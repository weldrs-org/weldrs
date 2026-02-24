mod common;

use polars::prelude::*;
use weldrs::prelude::*;

#[test]
fn test_save_and_reload_identical_predictions() {
    let df = common::make_test_df();
    let linker = common::make_trained_linker(&df);

    // Predict with original linker
    let predictions1 = linker
        .predict(&df.clone().lazy(), None)
        .unwrap()
        .collect()
        .unwrap();

    // Save and reload
    let json = linker.save_settings_json().unwrap();
    let restored = Linker::load_settings_json(&json).unwrap();

    // Predict with restored linker
    let predictions2 = restored
        .predict(&df.clone().lazy(), None)
        .unwrap()
        .collect()
        .unwrap();

    // Same number of predictions
    assert_eq!(predictions1.height(), predictions2.height());

    // Same match probabilities (sort both by uid pair for stable comparison)
    let sort_opts = SortMultipleOptions::new().with_order_descending(false);
    let sorted1 = predictions1
        .sort(["unique_id_l", "unique_id_r"], sort_opts.clone())
        .unwrap();
    let sorted2 = predictions2
        .sort(["unique_id_l", "unique_id_r"], sort_opts)
        .unwrap();

    let probs1: Vec<Option<f64>> = sorted1
        .column("match_probability")
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .collect();
    let probs2: Vec<Option<f64>> = sorted2
        .column("match_probability")
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .collect();

    assert_eq!(probs1.len(), probs2.len());
    for (p1, p2) in probs1.iter().zip(probs2.iter()) {
        match (p1, p2) {
            (Some(a), Some(b)) => {
                if a.is_nan() && b.is_nan() {
                    continue;
                }
                assert!(
                    (a - b).abs() < 1e-10,
                    "Probabilities should match: {a} vs {b}"
                );
            }
            (None, None) => {}
            _ => panic!("Mismatched null/non-null probabilities"),
        }
    }
}

#[test]
fn test_serialized_json_is_valid() {
    let df = common::make_test_df();
    let linker = common::make_trained_linker(&df);

    let json = linker.save_settings_json().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    // Check top-level keys are present
    assert!(parsed.get("link_type").is_some());
    assert!(parsed.get("comparisons").is_some());
    assert!(parsed.get("blocking_rules").is_some());
    assert!(parsed.get("probability_two_random_records_match").is_some());
    assert!(parsed.get("unique_id_column").is_some());
    assert!(parsed.get("training").is_some());

    // Comparisons should be an array
    let comparisons = parsed.get("comparisons").unwrap().as_array().unwrap();
    assert_eq!(comparisons.len(), 3);

    // Blocking rules should be an array
    let rules = parsed.get("blocking_rules").unwrap().as_array().unwrap();
    assert_eq!(rules.len(), 1);
}
