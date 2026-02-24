mod common;

use polars::prelude::*;

#[test]
fn test_full_pipeline_converges() {
    let df = common::make_test_df();
    let linker = common::make_trained_linker(&df);

    // Predict
    let predictions = linker
        .predict(&df.clone().lazy(), None)
        .unwrap()
        .collect()
        .unwrap();

    let col_names: Vec<&str> = predictions
        .get_column_names()
        .into_iter()
        .map(|s| s.as_str())
        .collect();
    assert!(col_names.contains(&"match_weight"));
    assert!(col_names.contains(&"match_probability"));
    assert!(predictions.height() > 0);

    // Cluster
    let clusters = linker
        .cluster_pairwise_predictions(&predictions, 0.5)
        .unwrap();
    assert!(clusters.height() > 0);
    let cluster_cols: Vec<&str> = clusters
        .get_column_names()
        .into_iter()
        .map(|s| s.as_str())
        .collect();
    assert!(cluster_cols.contains(&"unique_id"));
    assert!(cluster_cols.contains(&"cluster_id"));
}

#[test]
fn test_known_duplicates_score_high() {
    let df = common::make_test_df();
    let linker = common::make_trained_linker(&df);

    let predictions = linker
        .predict(&df.clone().lazy(), None)
        .unwrap()
        .collect()
        .unwrap();

    let uid_l = predictions.column("unique_id_l").unwrap().i64().unwrap();
    let uid_r = predictions.column("unique_id_r").unwrap().i64().unwrap();
    let probs = predictions
        .column("match_probability")
        .unwrap()
        .f64()
        .unwrap();

    // Check known duplicate pairs score relatively high
    // (With only 10 records and limited training, we use relaxed thresholds)
    let mut found_high_score_pair = false;
    for ((l, r), p) in uid_l
        .into_iter()
        .zip(uid_r.into_iter())
        .zip(probs.into_iter())
    {
        if let (Some(l), Some(r), Some(prob)) = (l, r, p) {
            // Known duplicates: (1,6) Smith, (2,7) Doe, (3,8) Williams
            let is_known_dup = matches!((l, r), (1, 6) | (2, 7) | (3, 8));
            if is_known_dup && prob > 0.3 {
                found_high_score_pair = true;
            }
            // Clearly non-matching pairs should score lower than known duplicates
            let is_clearly_non_match = matches!((l, r), (1, 4) | (1, 5) | (4, 9));
            if is_clearly_non_match {
                assert!(
                    prob < 0.5,
                    "Non-matching pair ({l},{r}) should score < 0.5, got {prob}"
                );
            }
        }
    }
    assert!(
        found_high_score_pair,
        "At least one known duplicate pair should score > 0.3"
    );
}

#[test]
fn test_clustering_groups_duplicates() {
    let df = common::make_test_df();
    let linker = common::make_trained_linker(&df);

    let predictions = linker
        .predict(&df.clone().lazy(), None)
        .unwrap()
        .collect()
        .unwrap();

    let clusters = linker
        .cluster_pairwise_predictions(&predictions, 0.3)
        .unwrap();

    let ids = clusters.column("unique_id").unwrap().i64().unwrap();
    let cids = clusters.column("cluster_id").unwrap().i64().unwrap();

    let id_to_cluster: std::collections::HashMap<i64, i64> = ids
        .into_no_null_iter()
        .zip(cids.into_no_null_iter())
        .collect();

    // IDs 1 and 6 (John/Jon Smith) should share a cluster if both present
    if id_to_cluster.contains_key(&1) && id_to_cluster.contains_key(&6) {
        assert_eq!(
            id_to_cluster[&1], id_to_cluster[&6],
            "IDs 1 and 6 should be in the same cluster"
        );
    }

    // IDs 3 and 8 (Bob/Robert Williams) should share a cluster if both present
    if id_to_cluster.contains_key(&3) && id_to_cluster.contains_key(&8) {
        assert_eq!(
            id_to_cluster[&3], id_to_cluster[&8],
            "IDs 3 and 8 should be in the same cluster"
        );
    }
}
