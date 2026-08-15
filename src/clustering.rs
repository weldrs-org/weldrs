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
use rustc_hash::{FxHashMap, FxHashSet};

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
        .map_err(|e| WeldrsError::Training {
            stage: "clustering",
            message: format!("Missing match_probability: {e}"),
        })?;
    let match_probs = mp.f64().map_err(|e| WeldrsError::Training {
        stage: "clustering",
        message: format!("match_probability type error: {e}"),
    })?;

    let uid_l_series = predictions
        .column(unique_id_l_col)
        .map_err(|e| WeldrsError::Training {
            stage: "clustering",
            message: format!("Missing {unique_id_l_col}: {e}"),
        })?;
    let uid_r_series = predictions
        .column(unique_id_r_col)
        .map_err(|e| WeldrsError::Training {
            stage: "clustering",
            message: format!("Missing {unique_id_r_col}: {e}"),
        })?;

    // We work with i64 IDs. If the column is a different integer type, cast it.
    let uid_l = uid_l_series
        .cast(&DataType::Int64)
        .map_err(|e| WeldrsError::Training {
            stage: "clustering",
            message: format!("Cannot cast {unique_id_l_col} to i64: {e}"),
        })?;
    let uid_r = uid_r_series
        .cast(&DataType::Int64)
        .map_err(|e| WeldrsError::Training {
            stage: "clustering",
            message: format!("Cannot cast {unique_id_r_col} to i64: {e}"),
        })?;
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
        .zip(uid_r_ca)
        .zip(match_probs)
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
    .map_err(|e| WeldrsError::Training {
        stage: "clustering",
        message: format!("Failed to build cluster DataFrame: {e}"),
    })?;

    Ok(df)
}

/// Cluster predictions with a per-source cardinality constraint: each cluster
/// holds at most one record from any given source dataset.
///
/// Edges (pairs with `match_probability >= threshold`) are processed greedily
/// in descending score order. Two records (or clusters) are only merged if the
/// resulting cluster would not contain two records from the same source — this
/// produces the "single best link" between datasets, the standard choice when
/// each source is internally de-duplicated. Ties are broken by `(uid_l, uid_r)`
/// for determinism.
///
/// Returns a DataFrame with columns `[unique_id, cluster_id]`.
///
/// # Errors
///
/// Returns an error if required columns are missing or the unique-id columns
/// cannot be cast to `i64`.
pub fn cluster_using_single_best_links(
    predictions: &DataFrame,
    threshold: f64,
    unique_id_l_col: &str,
    unique_id_r_col: &str,
    source_l_col: &str,
    source_r_col: &str,
) -> Result<DataFrame> {
    let col_f64 = |name: &str| -> Result<&Column> {
        predictions.column(name).map_err(|e| WeldrsError::Training {
            stage: "clustering",
            message: format!("Missing {name}: {e}"),
        })
    };

    let mp = col_f64("match_probability")?
        .f64()
        .map_err(|e| WeldrsError::Training {
            stage: "clustering",
            message: format!("match_probability type error: {e}"),
        })?;

    let uid_l = col_f64(unique_id_l_col)?
        .cast(&DataType::Int64)
        .map_err(|e| WeldrsError::Training {
            stage: "clustering",
            message: format!("Cannot cast {unique_id_l_col} to i64: {e}"),
        })?;
    let uid_r = col_f64(unique_id_r_col)?
        .cast(&DataType::Int64)
        .map_err(|e| WeldrsError::Training {
            stage: "clustering",
            message: format!("Cannot cast {unique_id_r_col} to i64: {e}"),
        })?;
    let uid_l_ca = uid_l.i64().unwrap();
    let uid_r_ca = uid_r.i64().unwrap();

    let src_l = col_f64(source_l_col)?
        .cast(&DataType::String)
        .map_err(|e| WeldrsError::Training {
            stage: "clustering",
            message: format!("Cannot cast {source_l_col} to string: {e}"),
        })?;
    let src_r = col_f64(source_r_col)?
        .cast(&DataType::String)
        .map_err(|e| WeldrsError::Training {
            stage: "clustering",
            message: format!("Cannot cast {source_r_col} to string: {e}"),
        })?;
    let src_l_ca = src_l.str().unwrap();
    let src_r_ca = src_r.str().unwrap();

    // Collect edges above threshold as (score, uid_l, uid_r, src_l, src_r).
    let mut edges: Vec<(f64, i64, i64, String, String)> = Vec::new();
    for i in 0..predictions.height() {
        let (p, l, r, sl, sr) = (
            mp.get(i),
            uid_l_ca.get(i),
            uid_r_ca.get(i),
            src_l_ca.get(i),
            src_r_ca.get(i),
        );
        if let (Some(p), Some(l), Some(r), Some(sl), Some(sr)) = (p, l, r, sl, sr)
            && p >= threshold
        {
            edges.push((p, l, r, sl.to_string(), sr.to_string()));
        }
    }
    // Descending by score; deterministic tie-break.
    edges.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });

    // Intern a record id, registering its source dataset on first sight.
    fn intern(
        id: i64,
        src: &str,
        id_to_index: &mut FxHashMap<i64, usize>,
        ids: &mut Vec<i64>,
        uf: &mut UnionFind,
        sources: &mut FxHashMap<usize, FxHashSet<String>>,
    ) -> usize {
        *id_to_index.entry(id).or_insert_with(|| {
            ids.push(id);
            let idx = uf.grow();
            let mut s = FxHashSet::default();
            s.insert(src.to_string());
            sources.insert(idx, s);
            idx
        })
    }

    let mut id_to_index: FxHashMap<i64, usize> = FxHashMap::default();
    let mut ids: Vec<i64> = Vec::new();
    let mut uf = UnionFind::new(0);
    // Per-cluster-root set of source datasets currently present.
    let mut sources: FxHashMap<usize, FxHashSet<String>> = FxHashMap::default();

    for (_score, l, r, sl, sr) in &edges {
        let il = intern(*l, sl, &mut id_to_index, &mut ids, &mut uf, &mut sources);
        let ir = intern(*r, sr, &mut id_to_index, &mut ids, &mut uf, &mut sources);
        let ra = uf.find(il);
        let rb = uf.find(ir);
        if ra == rb {
            continue;
        }
        // Merge only if the two clusters share no source dataset.
        let disjoint = {
            let sa = &sources[&ra];
            let sb = &sources[&rb];
            sa.is_disjoint(sb)
        };
        if disjoint {
            let set_a = sources.remove(&ra).unwrap_or_default();
            let set_b = sources.remove(&rb).unwrap_or_default();
            uf.union(ra, rb);
            let new_root = uf.find(ra);
            let mut merged = set_a;
            merged.extend(set_b);
            sources.insert(new_root, merged);
        }
    }

    let mut out_ids = Vec::with_capacity(ids.len());
    let mut out_clusters = Vec::with_capacity(ids.len());
    for (i, &id) in ids.iter().enumerate() {
        let root = uf.find(i);
        out_ids.push(id);
        out_clusters.push(ids[root]);
    }

    let n = out_ids.len();
    DataFrame::new(
        n,
        vec![
            Column::new("unique_id".into(), &out_ids),
            Column::new("cluster_id".into(), &out_clusters),
        ],
    )
    .map_err(|e| WeldrsError::Training {
        stage: "clustering",
        message: format!("Failed to build cluster DataFrame: {e}"),
    })
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
    fn test_single_best_links_respects_source_cardinality() {
        // Dataset a has records 1,2; dataset b has record 101. Both a-records
        // link to b's record, but a cluster may hold at most one record per
        // source, so only the strongest link (1,101) survives.
        let preds = df!(
            "unique_id_l" => [1i64, 2],
            "unique_id_r" => [101i64, 101],
            "source_dataset_l" => ["a", "a"],
            "source_dataset_r" => ["b", "b"],
            "match_probability" => [0.9, 0.8],
        )
        .unwrap();

        let clusters = cluster_using_single_best_links(
            &preds,
            0.5,
            "unique_id_l",
            "unique_id_r",
            "source_dataset_l",
            "source_dataset_r",
        )
        .unwrap();

        let ids = clusters.column("unique_id").unwrap().i64().unwrap();
        let cids = clusters.column("cluster_id").unwrap().i64().unwrap();
        let map: HashMap<i64, i64> = ids
            .into_no_null_iter()
            .zip(cids.into_no_null_iter())
            .collect();

        assert_eq!(map[&1], map[&101], "strongest link should cluster together");
        assert_ne!(
            map[&2], map[&101],
            "second a-record must not join the same cluster (source-cardinality)"
        );
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
