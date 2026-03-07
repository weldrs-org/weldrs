//! Shared test helpers available to unit tests across all modules.

use polars::prelude::*;

use crate::comparison::ComparisonBuilder;

/// Build a 10-record test DataFrame with known duplicate clusters.
///
/// Cluster A: ids 1, 6 ("John"/"Jon" Smith, London, same email)
/// Cluster B: ids 2, 7, 10 ("Jane"/"Janet"/"Jane" Doe, Manchester/"Manchster")
/// Cluster C: ids 3, 8 ("Bob"/"Robert" Williams, Bristol, same email)
/// Singletons: 4, 5, 9
pub fn make_test_df() -> DataFrame {
    df!(
        "unique_id" => [1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "first_name" => ["John", "Jane", "Bob", "Alice", "Eve", "Jon", "Janet", "Robert", "Charlie", "Jane"],
        "last_name" => ["Smith", "Doe", "Williams", "Brown", "Davis", "Smith", "Doe", "Williams", "Wilson", "Doe"],
        "city" => ["London", "Manchester", "Bristol", "Leeds", "York", "London", "Manchester", "Bristol", "Oxford", "Manchster"],
        "email" => [Some("john@example.com"), Some("jane@example.com"), Some("bob@example.com"),
                    Some("alice@example.com"), None, Some("john@example.com"), Some("janet@example.com"),
                    Some("bob@example.com"), None, Some("jane.doe@example.com")]
    )
    .unwrap()
}

/// Build a simple exact-match comparison: null + exact + else.
pub fn exact_match_comparison(col_name: &str) -> crate::comparison::Comparison {
    ComparisonBuilder::new(col_name)
        .null_level()
        .exact_match_level()
        .else_level()
        .build()
}

/// Build a fuzzy comparison: null + exact + jaro-winkler + else.
pub fn fuzzy_comparison(col_name: &str, jw_threshold: f64) -> crate::comparison::Comparison {
    ComparisonBuilder::new(col_name)
        .null_level()
        .exact_match_level()
        .jaro_winkler_level(jw_threshold)
        .else_level()
        .build()
}

/// Build a small paired DataFrame with `_l`/`_r` columns for testing expressions.
pub fn make_paired_df(
    uid_l: &[i64],
    uid_r: &[i64],
    col_name: &str,
    vals_l: &[&str],
    vals_r: &[&str],
) -> DataFrame {
    let col_l = format!("{col_name}_l");
    let col_r = format!("{col_name}_r");
    df!(
        "unique_id_l" => uid_l,
        "unique_id_r" => uid_r,
        &col_l => vals_l,
        &col_r => vals_r,
    )
    .unwrap()
}
