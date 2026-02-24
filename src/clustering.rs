//! Connected-components clustering via union-find.
//!
//! Pairwise predictions above a threshold are treated as edges in a graph.
//! This module groups records into clusters using a union-find (disjoint set)
//! data structure with path compression and union by rank.

use polars::prelude::*;
use std::collections::HashMap;

use crate::error::{Result, WeldrsError};

/// Union-Find (disjoint set) data structure with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}

/// Cluster pairwise predictions into groups of linked records using connected
/// components (union-find).
///
/// `predictions` should contain columns `unique_id_l` and `unique_id_r` (or
/// whatever the unique ID column is named with `_l` / `_r` suffixes), plus a
/// `match_probability` column.
///
/// Returns a DataFrame with columns `[unique_id, cluster_id]`.
pub fn cluster_pairwise_predictions(
    predictions: &DataFrame,
    threshold: f64,
    unique_id_l_col: &str,
    unique_id_r_col: &str,
) -> Result<DataFrame> {
    let mp = predictions
        .column("match_probability")
        .map_err(|e| WeldrsError::Training(format!("Missing match_probability: {e}")))?;
    let match_probs = mp
        .f64()
        .map_err(|e| WeldrsError::Training(format!("match_probability type error: {e}")))?;

    let uid_l_series = predictions
        .column(unique_id_l_col)
        .map_err(|e| WeldrsError::Training(format!("Missing {unique_id_l_col}: {e}")))?;
    let uid_r_series = predictions
        .column(unique_id_r_col)
        .map_err(|e| WeldrsError::Training(format!("Missing {unique_id_r_col}: {e}")))?;

    // We work with i64 IDs. If the column is a different integer type, cast it.
    let uid_l = uid_l_series
        .cast(&DataType::Int64)
        .map_err(|e| WeldrsError::Training(format!("Cannot cast {unique_id_l_col} to i64: {e}")))?;
    let uid_r = uid_r_series
        .cast(&DataType::Int64)
        .map_err(|e| WeldrsError::Training(format!("Cannot cast {unique_id_r_col} to i64: {e}")))?;
    let uid_l_ca = uid_l.i64().unwrap();
    let uid_r_ca = uid_r.i64().unwrap();

    // Collect all unique IDs and build an index map.
    let mut id_to_index: HashMap<i64, usize> = HashMap::new();
    let mut ids: Vec<i64> = Vec::new();

    for (l, r, mp) in uid_l_ca
        .into_iter()
        .zip(uid_r_ca.into_iter())
        .zip(match_probs.into_iter())
        .map(|((l, r), mp)| (l, r, mp))
    {
        if let (Some(l), Some(r), Some(p)) = (l, r, mp)
            && p >= threshold
        {
            if let std::collections::hash_map::Entry::Vacant(e) = id_to_index.entry(l) {
                e.insert(ids.len());
                ids.push(l);
            }
            if let std::collections::hash_map::Entry::Vacant(e) = id_to_index.entry(r) {
                e.insert(ids.len());
                ids.push(r);
            }
        }
    }

    // Build union-find and process edges.
    let mut uf = UnionFind::new(ids.len());

    for (l, r, mp) in uid_l_ca
        .into_iter()
        .zip(uid_r_ca.into_iter())
        .zip(match_probs.into_iter())
        .map(|((l, r), mp)| (l, r, mp))
    {
        if let (Some(l), Some(r), Some(p)) = (l, r, mp)
            && p >= threshold
        {
            let il = id_to_index[&l];
            let ir = id_to_index[&r];
            uf.union(il, ir);
        }
    }

    // Build output: [unique_id, cluster_id].
    let mut out_ids = Vec::with_capacity(ids.len());
    let mut out_clusters = Vec::with_capacity(ids.len());

    for (i, &id) in ids.iter().enumerate() {
        let root = uf.find(i);
        out_ids.push(id);
        out_clusters.push(ids[root]);
    }

    let df = DataFrame::new(vec![
        Column::new("unique_id".into(), &out_ids),
        Column::new("cluster_id".into(), &out_clusters),
    ])
    .map_err(|e| WeldrsError::Training(format!("Failed to build cluster DataFrame: {e}")))?;

    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn predictions_df(uid_l: &[i64], uid_r: &[i64], probs: &[f64]) -> DataFrame {
        df!(
            "unique_id_l" => uid_l,
            "unique_id_r" => uid_r,
            "match_probability" => probs,
        )
        .unwrap()
    }

    #[test]
    fn test_transitive_closure() {
        // (1,2) + (2,3) → cluster {1,2,3}
        let preds = predictions_df(&[1, 2], &[2, 3], &[0.9, 0.9]);
        let clusters =
            cluster_pairwise_predictions(&preds, 0.5, "unique_id_l", "unique_id_r").unwrap();

        assert_eq!(clusters.height(), 3);

        let cids = clusters.column("cluster_id").unwrap().i64().unwrap();

        // All three should share the same cluster_id
        let cluster_ids: std::collections::HashSet<i64> = cids.into_no_null_iter().collect();
        assert_eq!(cluster_ids.len(), 1, "All should be in one cluster");
    }

    #[test]
    fn test_disjoint_clusters() {
        // (1,2) + (3,4) → two separate clusters
        let preds = predictions_df(&[1, 3], &[2, 4], &[0.9, 0.9]);
        let clusters =
            cluster_pairwise_predictions(&preds, 0.5, "unique_id_l", "unique_id_r").unwrap();

        assert_eq!(clusters.height(), 4);

        let ids = clusters.column("unique_id").unwrap().i64().unwrap();
        let cids = clusters.column("cluster_id").unwrap().i64().unwrap();

        // Find cluster IDs for ids 1 and 3
        let id_cluster: HashMap<i64, i64> = ids
            .into_no_null_iter()
            .zip(cids.into_no_null_iter())
            .collect();

        assert_eq!(id_cluster[&1], id_cluster[&2]);
        assert_eq!(id_cluster[&3], id_cluster[&4]);
        assert_ne!(id_cluster[&1], id_cluster[&3]);
    }

    #[test]
    fn test_threshold_filtering() {
        // (1,2) at 0.9 is above threshold, (2,3) at 0.3 is below
        let preds = predictions_df(&[1, 2], &[2, 3], &[0.9, 0.3]);
        let clusters =
            cluster_pairwise_predictions(&preds, 0.5, "unique_id_l", "unique_id_r").unwrap();

        // Only pair (1,2) survives → cluster of 2 IDs, id 3 is excluded
        assert_eq!(clusters.height(), 2);

        let ids: Vec<i64> = clusters
            .column("unique_id")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&3));
    }

    #[test]
    fn test_single_edge_cluster() {
        let preds = predictions_df(&[10], &[20], &[0.95]);
        let clusters =
            cluster_pairwise_predictions(&preds, 0.5, "unique_id_l", "unique_id_r").unwrap();

        assert_eq!(clusters.height(), 2);
    }

    #[test]
    fn test_empty_predictions() {
        let preds = predictions_df(&[], &[], &[]);
        let clusters =
            cluster_pairwise_predictions(&preds, 0.5, "unique_id_l", "unique_id_r").unwrap();

        assert_eq!(clusters.height(), 0);
    }
}
