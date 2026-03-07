#![allow(dead_code)]

use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::prelude::*;

/// Build a 10-record test DataFrame with known duplicate clusters.
pub fn make_test_df() -> DataFrame {
    df!(
        "unique_id" => [1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "first_name" => ["John", "Jane", "Bob", "Alice", "Eve", "Jon", "Janet", "Robert", "Charlie", "Jane"],
        "surname" => ["Smith", "Doe", "Williams", "Brown", "Davis", "Smith", "Doe", "Williams", "Wilson", "Doe"],
        "city" => ["London", "Manchester", "Bristol", "Leeds", "York", "London", "Manchester", "Bristol", "Oxford", "Manchster"],
        "email" => [Some("john@example.com"), Some("jane@example.com"), Some("bob@example.com"),
                    Some("alice@example.com"), None, Some("john@example.com"), Some("janet@example.com"),
                    Some("bob@example.com"), None, Some("jane.doe@example.com")]
    )
    .unwrap()
}

/// Build an exact-match comparison: null + exact + else.
pub fn exact_match_comparison(col_name: &str) -> Comparison {
    ComparisonBuilder::new(col_name)
        .null_level()
        .exact_match_level()
        .else_level()
        .build()
        .expect("test helper: exact_match_comparison should always be valid")
}

/// Build a fuzzy comparison: null + exact + jaro-winkler + else.
pub fn fuzzy_comparison(col_name: &str, jw_threshold: f64) -> Comparison {
    ComparisonBuilder::new(col_name)
        .null_level()
        .exact_match_level()
        .jaro_winkler_level(jw_threshold)
        .else_level()
        .build()
        .expect("test helper: fuzzy_comparison should always be valid")
}

/// Build a fully trained Linker from the test data.
pub fn make_trained_linker(df: &DataFrame) -> Linker {
    let settings = Settings::builder(LinkType::DedupeOnly)
        .comparison(fuzzy_comparison("first_name", 0.85))
        .comparison(exact_match_comparison("surname"))
        .comparison(exact_match_comparison("city"))
        .blocking_rule(BlockingRule::on(&["surname"]))
        .build()
        .unwrap();

    let mut linker = Linker::new(settings).unwrap();
    let lf = df.clone().lazy();

    // Estimate lambda
    linker
        .estimate_probability_two_random_records_match(
            &lf,
            &[BlockingRule::on(&["first_name", "surname"])],
            1.0,
        )
        .unwrap();

    // Estimate u values
    linker.estimate_u_using_random_sampling(&lf, 200).unwrap();

    // EM training session 1: block on surname
    linker
        .estimate_parameters_using_em(&lf, &BlockingRule::on(&["surname"]))
        .unwrap();

    // EM training session 2: block on city
    linker
        .estimate_parameters_using_em(&lf, &BlockingRule::on(&["city"]))
        .unwrap();

    linker
}
