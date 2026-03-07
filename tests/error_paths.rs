//! Tests for error handling and edge cases.

use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::prelude::*;

// ── ComparisonBuilder validation errors ──────────────────────────────

#[test]
fn test_comparison_builder_no_levels_errors() {
    let result = ComparisonBuilder::new("name").build();
    assert!(result.is_err());
}

#[test]
fn test_comparison_builder_missing_else_errors() {
    let result = ComparisonBuilder::new("name")
        .null_level()
        .exact_match_level()
        .build();
    assert!(result.is_err());
}

#[test]
fn test_comparison_builder_else_not_last_errors() {
    let result = ComparisonBuilder::new("name")
        .null_level()
        .else_level()
        .exact_match_level()
        .build();
    assert!(result.is_err());
}

// ── SettingsBuilder validation errors ────────────────────────────────

#[test]
fn test_settings_builder_no_comparisons_errors() {
    let result = Settings::builder(LinkType::DedupeOnly).build();
    assert!(result.is_err());
}

// ── Linker error paths ──────────────────────────────────────────────

#[test]
fn test_estimate_lambda_empty_rules_errors() {
    let comp = ComparisonBuilder::new("name")
        .null_level()
        .exact_match_level()
        .else_level()
        .build()
        .unwrap();

    let settings = Settings::builder(LinkType::DedupeOnly)
        .comparison(comp)
        .build()
        .unwrap();

    let mut linker = Linker::new(settings).unwrap();
    let df = df!(
        "unique_id" => [1i64, 2, 3],
        "name" => ["Alice", "Bob", "Charlie"],
    )
    .unwrap();

    let result = linker.estimate_probability_two_random_records_match(
        &df.lazy(),
        &[], // empty rules
        1.0,
    );
    assert!(result.is_err(), "Empty deterministic rules should error");
}

#[test]
fn test_estimate_lambda_tiny_df_errors() {
    let comp = ComparisonBuilder::new("name")
        .null_level()
        .exact_match_level()
        .else_level()
        .build()
        .unwrap();

    let settings = Settings::builder(LinkType::DedupeOnly)
        .comparison(comp)
        .build()
        .unwrap();

    let mut linker = Linker::new(settings).unwrap();
    let df = df!(
        "unique_id" => [1i64],
        "name" => ["Alice"],
    )
    .unwrap();

    let result = linker.estimate_probability_two_random_records_match(
        &df.lazy(),
        &[BlockingRule::on(&["name"])],
        1.0,
    );
    assert!(result.is_err(), "Fewer than 2 records should error");
}

// ── Clustering error paths ──────────────────────────────────────────

#[test]
fn test_cluster_missing_probability_column_errors() {
    // DataFrame without match_probability
    let df = df!(
        "unique_id_l" => [1i64],
        "unique_id_r" => [2i64],
    )
    .unwrap();

    let result =
        weldrs::clustering::cluster_pairwise_predictions(&df, 0.5, "unique_id_l", "unique_id_r");
    assert!(
        result.is_err(),
        "Missing match_probability column should error"
    );
}

#[test]
fn test_cluster_missing_uid_column_errors() {
    let df = df!(
        "match_probability" => [0.9],
    )
    .unwrap();

    let result =
        weldrs::clustering::cluster_pairwise_predictions(&df, 0.5, "unique_id_l", "unique_id_r");
    assert!(result.is_err(), "Missing uid column should error");
}

// ── Explain error paths ─────────────────────────────────────────────

#[test]
fn test_explain_pair_out_of_bounds_errors() {
    let comp = ComparisonBuilder::new("name")
        .null_level()
        .exact_match_level()
        .else_level()
        .build()
        .unwrap();

    let settings = Settings::builder(LinkType::DedupeOnly)
        .comparison(comp)
        .blocking_rule(BlockingRule::on(&["name"]))
        .build()
        .unwrap();

    let linker = Linker::new(settings).unwrap();

    let predictions = df!(
        "unique_id_l" => [1i64],
        "unique_id_r" => [2i64],
        "gamma_name" => [1i8],
        "bf_name" => [9.0],
        "match_weight" => [5.0],
        "match_probability" => [0.9],
    )
    .unwrap();

    // row_index 99 is out of bounds for a 1-row DataFrame
    let result = linker.explain_pair(&predictions, 99);
    assert!(result.is_err(), "Out of bounds row_index should error");
}

// ── Serialization error paths ───────────────────────────────────────

#[test]
fn test_load_settings_invalid_json_errors() {
    let result = Linker::load_settings_json("not valid json at all {{{");
    assert!(result.is_err(), "Invalid JSON should error");
}

#[test]
fn test_load_settings_wrong_schema_errors() {
    let result = Linker::load_settings_json(r#"{"key": "value"}"#);
    assert!(result.is_err(), "Wrong JSON schema should error");
}
