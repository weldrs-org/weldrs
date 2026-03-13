//! Connected-components clustering via union-find.
//!
//! Pairwise predictions above a threshold are treated as edges in a graph.
//! This module groups records into clusters using a union-find (disjoint set)
//! data structure with path compression and union by rank.
//!
//! This is **step 5** of the pipeline — after [`predict`](crate::predict)
//! scores candidate pairs, this module groups them into clusters of linked
//! records.
//!
//! # Example
//!
//! ```no_run
//! use polars::prelude::*;
//! use weldrs::clustering::cluster_pairwise_predictions;
//!
//! // `predictions` is a DataFrame with unique_id_l, unique_id_r,
//! // and match_probability columns (output of predict).
//! # let predictions = DataFrame::empty();
//! let clusters = cluster_pairwise_predictions(
//!     &predictions,
//!     0.5,             // threshold — only pairs at or above this probability
//!     "unique_id_l",
//!     "unique_id_r",
//! ).unwrap();
//! // Returns a DataFrame with [unique_id, cluster_id] columns.
//! ```

use polars::prelude::*;
use rustc_hash::FxHashMap;

use crate::error::{Result, WeldrsError};

/// Union-Find (disjoint set) data structure with iterative path compression
/// and union by rank. Supports incremental growth via `grow()`.
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

    /// Add one new element (returns its index).
    fn grow(&mut self) -> usize {
        let idx = self.parent.len();
        self.parent.push(idx);
        self.rank.push(0);
        idx
    }

    /// Iterative path-compression find — avoids stack overflow on deep chains.
    fn find(&mut self, mut x: usize) -> usize {
        // First pass: find root.
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Second pass: compress path.
        while self.parent[x] != root {
            let next = self.parent[x];
            self.parent[x] = root;
            x = next;
        }
        root
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
///
/// # Errors
///
/// Returns an error if the predictions DataFrame is missing required
/// columns (`match_probability`, or the specified unique ID columns),
/// or if the unique ID columns cannot be cast to `i64`.
///
/// # Examples
///
/// ```no_run
/// # use polars::prelude::*;
/// use weldrs::clustering::cluster_pairwise_predictions;
///
/// # let predictions = DataFrame::empty();
/// let clusters = cluster_pairwise_predictions(
///     &predictions,
///     0.5,
///     "unique_id_l",
///     "unique_id_r",
/// ).unwrap();
/// ```
pub fn cluster_pairwise_predictions(
    predictions: &DataFrame,
    threshold: f64,
    unique_id_l_col: &str,
    unique_id_r_col: &str,
) -> Result<DataFrame> {
    let mp = predictions
        .column("match_probability")
        .map_err(|e| WeldrsError::Training { stage: "clustering", message: format!("Missing match_probability: {e}") })?;
    let match_probs = mp
        .f64()
        .map_err(|e| WeldrsError::Training { stage: "clustering", message: format!("match_probability type error: {e}") })?;

    let uid_l_series = predictions
        .column(unique_id_l_col)
        .map_err(|e| WeldrsError::Training { stage: "clustering", message: format!("Missing {unique_id_l_col}: {e}") })?;
    let uid_r_series = predictions
        .column(unique_id_r_col)
        .map_err(|e| WeldrsError::Training { stage: "clustering", message: format!("Missing {unique_id_r_col}: {e}") })?;

    // We work with i64 IDs. If the column is a different integer type, cast it.
    let uid_l = uid_l_series
        .cast(&DataType::Int64)
        .map_err(|e| WeldrsError::Training { stage: "clustering", message: format!("Cannot cast {unique_id_l_col} to i64: {e}") })?;
    let uid_r = uid_r_series
        .cast(&DataType::Int64)
        .map_err(|e| WeldrsError::Training { stage: "clustering", message: format!("Cannot cast {unique_id_r_col} to i64: {e}") })?;
    let uid_l_ca = uid_l.i64().unwrap();
    let uid_r_ca = uid_r.i64().unwrap();

    // Single-pass: collect IDs and union in one iteration using a growable
    // union-find. FxHashMap is 2–5x faster than SipHash for integer keys.
    let n_estimate = predictions.height();
    let mut id_to_index: FxHashMap<i64, usize> =
        FxHashMap::with_capacity_and_hasher(n_estimate, Default::default());
    let mut ids: Vec<i64> = Vec::with_capacity(n_estimate);
    let mut uf = UnionFind::new(0);

    for (l, r, mp) in uid_l_ca
        .into_iter()
        .zip(uid_r_ca.into_iter())
        .zip(match_probs.into_iter())
        .map(|((l, r), mp)| (l, r, mp))
    {
        if let (Some(l), Some(r), Some(p)) = (l, r, mp)
            && p >= threshold
        {
            let il = *id_to_index.entry(l).or_insert_with(|| {
                ids.push(l);
                uf.grow()
            });
            let ir = *id_to_index.entry(r).or_insert_with(|| {
                ids.push(r);
                uf.grow()
            });
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

    let n = out_ids.len();
    let df = DataFrame::new(
        n,
        vec![
            Column::new("unique_id".into(), &out_ids),
            Column::new("cluster_id".into(), &out_clusters),
        ],
    )
    .map_err(|e| WeldrsError::Training { stage: "clustering", message: format!("Failed to build cluster DataFrame: {e}") })?;

    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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

    #[test]
    fn test_deep_chain_no_stack_overflow() {
        // Create a chain of 10K sequential unions: 0-1, 1-2, 2-3, ..., 9999-10000.
        let n = 10_000;
        let uid_l: Vec<i64> = (0..n).collect();
        let uid_r: Vec<i64> = (1..=n).collect();
        let probs: Vec<f64> = vec![0.9; n as usize];

        let preds = predictions_df(&uid_l, &uid_r, &probs);
        let clusters =
            cluster_pairwise_predictions(&preds, 0.5, "unique_id_l", "unique_id_r").unwrap();

        // All records should be in one cluster.
        let cids = clusters.column("cluster_id").unwrap().i64().unwrap();
        let cluster_ids: std::collections::HashSet<i64> = cids.into_no_null_iter().collect();
        assert_eq!(cluster_ids.len(), 1, "All should be in one cluster");
        assert_eq!(clusters.height(), (n + 1) as usize);
    }
}
