use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::prelude::*;

/// Eight records in two cities; `entity` is the ground-truth cluster id.
fn data() -> DataFrame {
    df!(
        "unique_id" => [1i64, 2, 3, 4, 5, 6, 7, 8],
        "entity"    => [10i64, 10, 20, 20, 30, 30, 40, 50],
        "first_name" => ["John", "Jon", "Mary", "Mary", "Sue", "Susan", "Bob", "Eve"],
        "city" => ["London", "London", "Paris", "Paris", "Paris", "Paris", "London", "London"],
    )
    .unwrap()
}

fn linker() -> Linker {
    let settings = Settings::builder(LinkType::DedupeOnly)
        .comparison(
            ComparisonBuilder::new("first_name")
                .null_level()
                .exact_match_level()
                .jaro_winkler_level(0.85)
                .else_level()
                .build()
                .unwrap(),
        )
        .blocking_rule(BlockingRule::on(&["city"]))
        .probability_two_random_records_match(0.2)
        .build()
        .unwrap();
    Linker::new(settings).unwrap()
}

#[test]
fn test_estimate_m_from_label_column_then_predict() {
    let mut linker = linker();
    let lf = data().lazy();

    // Train m from the ground-truth entity column, u from random sampling.
    linker.estimate_u_using_random_sampling(&lf, 200).unwrap();
    linker.estimate_m_from_label_column(&lf, "entity").unwrap();

    // m for the exact-match level should be the strongest signal.
    let comp = &linker.settings().comparisons[0];
    let exact = comp
        .comparison_levels
        .iter()
        .find(|l| l.comparison_vector_value == 2)
        .unwrap();
    assert!(
        exact.m_probability.unwrap() > 0.0,
        "exact-match m should be estimated from labelled matches"
    );

    let predictions = linker.predict(&lf, None).unwrap().collect().unwrap();
    assert!(predictions.height() > 0);
}

#[test]
fn test_linker_accuracy_analysis_end_to_end() {
    let mut linker = linker();
    let lf = data().lazy();
    linker.estimate_u_using_random_sampling(&lf, 200).unwrap();
    linker.estimate_m_from_label_column(&lf, "entity").unwrap();
    let predictions = linker.predict(&lf, None).unwrap().collect().unwrap();

    // Build a labelled-pair table from the entity column for the blocked pairs:
    // a predicted pair is a true match iff the two records share an entity.
    // Here we hand-label the within-city candidate pairs.
    let labels = df!(
        "unique_id_l" => [1i64, 3, 5, 3, 7],
        "unique_id_r" => [2i64, 4, 6, 5, 8],
        // (1,2) John/Jon same entity 10 → match; (3,4) Mary/Mary entity 20 → match;
        // (5,6) Sue/Susan entity 30 → match; (3,5) diff entity → non-match;
        // (7,8) Bob/Eve diff entity → non-match.
        "is_match" => [true, true, true, false, false],
    )
    .unwrap();

    let metrics = linker.accuracy_analysis(&predictions, &labels).unwrap();
    assert!(!metrics.is_empty());

    // There should be a threshold achieving perfect recall (all true matches
    // recovered) since matched pairs share first names closely.
    assert!(
        metrics.iter().any(|m| (m.recall - 1.0).abs() < 1e-9),
        "expected a threshold with full recall"
    );

    // ROC / PR tables come back well-formed.
    let roc = linker.roc_table(&predictions, &labels).unwrap();
    assert!(roc.height() > 0);
    let pr = linker.precision_recall_table(&predictions, &labels).unwrap();
    assert!(pr.height() > 0);
}
