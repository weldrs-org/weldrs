//! Graph metrics over the pairwise-prediction graph.
//!
//! Treating records as nodes and above-threshold predictions as edges, this
//! module computes node-, edge-, and cluster-level structural metrics that help
//! diagnose cluster quality:
//!
//! - **node degree** — how many edges touch a record;
//! - **edge bridges** — edges whose removal would split a cluster (a bridge is
//!   a weak point: the cluster hangs together only through that single link);
//! - **cluster density** — how close a cluster is to fully connected.
//!
//! Powered by [`petgraph`](https://docs.rs/petgraph).

use petgraph::algo::bridges;
use petgraph::graph::UnGraph;
use petgraph::unionfind::UnionFind;
use petgraph::visit::{EdgeRef, NodeIndexable};
use polars::prelude::*;
use rustc_hash::FxHashMap;

use crate::error::{Result, WeldrsError};

/// Node, edge, and cluster metrics for a prediction graph.
#[derive(Debug, Clone)]
pub struct GraphMetrics {
    /// One row per record: `[node_id, degree, cluster_id]`.
    pub nodes: DataFrame,
    /// One row per edge: `[unique_id_l, unique_id_r, is_bridge]`.
    pub edges: DataFrame,
    /// One row per cluster: `[cluster_id, n_nodes, n_edges, density]`.
    pub clusters: DataFrame,
}

/// Compute graph metrics from scored predictions.
///
/// Only pairs with `match_probability >= threshold` form edges. `cluster_id` is
/// the smallest record id in each connected component.
///
/// # Errors
///
/// Returns an error if required columns are missing or cannot be cast to `i64`,
/// or if building an output DataFrame fails.
pub fn compute_graph_metrics(
    predictions: &DataFrame,
    threshold: f64,
    unique_id_l_col: &str,
    unique_id_r_col: &str,
) -> Result<GraphMetrics> {
    let get = |name: &str| -> Result<&Column> {
        predictions.column(name).map_err(|e| WeldrsError::Training {
            stage: "graph_metrics",
            message: format!("Missing {name}: {e}"),
        })
    };

    let mp = get("match_probability")?
        .f64()
        .map_err(|e| WeldrsError::Training {
            stage: "graph_metrics",
            message: format!("match_probability type error: {e}"),
        })?;
    let uid_l =
        get(unique_id_l_col)?
            .cast(&DataType::Int64)
            .map_err(|e| WeldrsError::Training {
                stage: "graph_metrics",
                message: format!("Cannot cast {unique_id_l_col} to i64: {e}"),
            })?;
    let uid_r =
        get(unique_id_r_col)?
            .cast(&DataType::Int64)
            .map_err(|e| WeldrsError::Training {
                stage: "graph_metrics",
                message: format!("Cannot cast {unique_id_r_col} to i64: {e}"),
            })?;
    let uid_l_ca = uid_l.i64().unwrap();
    let uid_r_ca = uid_r.i64().unwrap();

    // Build the undirected graph, de-duplicating edges (canonical (min,max)).
    let mut graph: UnGraph<i64, ()> = UnGraph::new_undirected();
    let mut id_to_node: FxHashMap<i64, petgraph::graph::NodeIndex> = FxHashMap::default();
    let mut seen_edges: rustc_hash::FxHashSet<(i64, i64)> = rustc_hash::FxHashSet::default();
    // Preserve insertion order of unique edges for stable output.
    let mut edge_ids: Vec<(i64, i64)> = Vec::new();

    for i in 0..predictions.height() {
        if let (Some(p), Some(l), Some(r)) = (mp.get(i), uid_l_ca.get(i), uid_r_ca.get(i))
            && p >= threshold
            && l != r
        {
            let (a, b) = if l <= r { (l, r) } else { (r, l) };
            if !seen_edges.insert((a, b)) {
                continue;
            }
            let na = *id_to_node.entry(a).or_insert_with(|| graph.add_node(a));
            let nb = *id_to_node.entry(b).or_insert_with(|| graph.add_node(b));
            graph.add_edge(na, nb, ());
            edge_ids.push((a, b));
        }
    }

    // Connected-component labels via union-find over node indices.
    let mut uf = UnionFind::new(graph.node_bound());
    for e in graph.edge_references() {
        uf.union(e.source().index(), e.target().index());
    }
    // cluster_id = smallest original id in the component.
    let mut root_to_cluster: FxHashMap<usize, i64> = FxHashMap::default();
    for node in graph.node_indices() {
        let root = uf.find(node.index());
        let id = graph[node];
        root_to_cluster
            .entry(root)
            .and_modify(|c| {
                if id < *c {
                    *c = id;
                }
            })
            .or_insert(id);
    }

    // ── Node metrics ────────────────────────────────────────────────
    let mut node_id: Vec<i64> = Vec::with_capacity(graph.node_count());
    let mut degree: Vec<u32> = Vec::with_capacity(graph.node_count());
    let mut node_cluster: Vec<i64> = Vec::with_capacity(graph.node_count());
    for node in graph.node_indices() {
        node_id.push(graph[node]);
        degree.push(graph.edges(node).count() as u32);
        node_cluster.push(root_to_cluster[&uf.find(node.index())]);
    }

    // ── Edge metrics (bridges) ──────────────────────────────────────
    let mut bridge_set: rustc_hash::FxHashSet<(i64, i64)> = rustc_hash::FxHashSet::default();
    for e in bridges(&graph) {
        let a = graph[e.source()];
        let b = graph[e.target()];
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        bridge_set.insert((a, b));
    }
    let edge_l: Vec<i64> = edge_ids.iter().map(|(a, _)| *a).collect();
    let edge_r: Vec<i64> = edge_ids.iter().map(|(_, b)| *b).collect();
    let is_bridge: Vec<bool> = edge_ids.iter().map(|p| bridge_set.contains(p)).collect();

    // ── Cluster metrics ─────────────────────────────────────────────
    let mut cluster_nodes: FxHashMap<i64, u32> = FxHashMap::default();
    let mut cluster_edges: FxHashMap<i64, u32> = FxHashMap::default();
    for node in graph.node_indices() {
        let c = root_to_cluster[&uf.find(node.index())];
        *cluster_nodes.entry(c).or_insert(0) += 1;
    }
    for (a, _b) in &edge_ids {
        let c = root_to_cluster[&uf.find(id_to_node[a].index())];
        *cluster_edges.entry(c).or_insert(0) += 1;
    }
    let mut cluster_ids: Vec<i64> = cluster_nodes.keys().copied().collect();
    cluster_ids.sort_unstable();
    let mut c_id = Vec::with_capacity(cluster_ids.len());
    let mut c_nodes = Vec::with_capacity(cluster_ids.len());
    let mut c_edges = Vec::with_capacity(cluster_ids.len());
    let mut c_density = Vec::with_capacity(cluster_ids.len());
    for c in &cluster_ids {
        let n = cluster_nodes[c];
        let m = *cluster_edges.get(c).unwrap_or(&0);
        let density = if n > 1 {
            2.0 * m as f64 / (n as f64 * (n as f64 - 1.0))
        } else {
            0.0
        };
        c_id.push(*c);
        c_nodes.push(n);
        c_edges.push(m);
        c_density.push(density);
    }

    let nodes = DataFrame::new(
        node_id.len(),
        vec![
            Column::new("node_id".into(), node_id),
            Column::new("degree".into(), degree),
            Column::new("cluster_id".into(), node_cluster),
        ],
    )
    .map_err(WeldrsError::Polars)?;
    let edges = DataFrame::new(
        edge_l.len(),
        vec![
            Column::new("unique_id_l".into(), edge_l),
            Column::new("unique_id_r".into(), edge_r),
            Column::new("is_bridge".into(), is_bridge),
        ],
    )
    .map_err(WeldrsError::Polars)?;
    let clusters = DataFrame::new(
        c_id.len(),
        vec![
            Column::new("cluster_id".into(), c_id),
            Column::new("n_nodes".into(), c_nodes),
            Column::new("n_edges".into(), c_edges),
            Column::new("density".into(), c_density),
        ],
    )
    .map_err(WeldrsError::Polars)?;

    Ok(GraphMetrics {
        nodes,
        edges,
        clusters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_metrics_triangle_plus_tail() {
        // Triangle 1-2-3 (no bridges) plus a tail 3-4 (a bridge).
        let preds = df!(
            "unique_id_l" => [1i64, 2, 1, 3],
            "unique_id_r" => [2i64, 3, 3, 4],
            "match_probability" => [0.9, 0.9, 0.9, 0.9],
        )
        .unwrap();

        let m = compute_graph_metrics(&preds, 0.5, "unique_id_l", "unique_id_r").unwrap();

        // 4 nodes, all in one cluster.
        assert_eq!(m.nodes.height(), 4);
        assert_eq!(m.clusters.height(), 1);

        // Only the 3-4 edge is a bridge.
        let el = m.edges.column("unique_id_l").unwrap().i64().unwrap();
        let er = m.edges.column("unique_id_r").unwrap().i64().unwrap();
        let br = m.edges.column("is_bridge").unwrap().bool().unwrap();
        for ((l, r), b) in el
            .into_no_null_iter()
            .zip(er.into_no_null_iter())
            .zip(br.into_no_null_iter())
        {
            let expected = (l, r) == (3, 4);
            assert_eq!(b, expected, "edge ({l},{r}) bridge flag");
        }

        // Node 4 has degree 1; nodes 1,2,3 have degree 2.
        let nid = m.nodes.column("node_id").unwrap().i64().unwrap();
        let deg = m.nodes.column("degree").unwrap().u32().unwrap();
        let degree_of: std::collections::HashMap<i64, u32> = nid
            .into_no_null_iter()
            .zip(deg.into_no_null_iter())
            .collect();
        assert_eq!(degree_of[&4], 1);
        assert_eq!(degree_of[&1], 2);
    }

    #[test]
    fn test_graph_metrics_density() {
        // A full triangle: density should be 1.0.
        let preds = df!(
            "unique_id_l" => [1i64, 2, 1],
            "unique_id_r" => [2i64, 3, 3],
            "match_probability" => [0.9, 0.9, 0.9],
        )
        .unwrap();
        let m = compute_graph_metrics(&preds, 0.5, "unique_id_l", "unique_id_r").unwrap();
        let density = m.clusters.column("density").unwrap().f64().unwrap();
        assert!((density.get(0).unwrap() - 1.0).abs() < 1e-12);
    }
}
