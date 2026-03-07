mod common;

use polars::prelude::*;
use weldrs::prelude::*;

/// Build two small datasets that share some records for LinkOnly testing.
fn make_two_datasets() -> (DataFrame, DataFrame) {
    let ds_a = df!(
        "unique_id" => [1i64, 2, 3, 4, 5],
        "first_name" => ["John", "Jane", "Bob", "Alice", "Eve"],
        "surname" => ["Smith", "Doe", "Williams", "Brown", "Davis"],
        "city" => ["London", "Manchester", "Bristol", "Leeds", "York"],
    )
    .unwrap();

    let ds_b = df!(
        "unique_id" => [101i64, 102, 103, 104, 105],
        "first_name" => ["Jon", "Janet", "Robert", "Alice", "Charlie"],
        "surname" => ["Smith", "Doe", "Williams", "Brown", "Wilson"],
        "city" => ["London", "Manchester", "Bristol", "Leeds", "Oxford"],
    )
    .unwrap();

    (ds_a, ds_b)
}

/// Combine two datasets with a source column for LinkOnly.
fn combine_for_link_only(ds_a: &DataFrame, ds_b: &DataFrame) -> DataFrame {
    let n_a = ds_a.height();
    let n_b = ds_b.height();

    let source_a = Column::new("source_dataset".into(), &vec!["a"; n_a]);
    let source_b = Column::new("source_dataset".into(), &vec!["b"; n_b]);

    let mut a = ds_a.clone();
    a.with_column(source_a).unwrap();

    let mut b = ds_b.clone();
    b.with_column(source_b).unwrap();

    a.vstack(&b).unwrap()
}

#[test]
fn test_link_only_pipeline_produces_pairs() {
    let (ds_a, ds_b) = make_two_datasets();
    let combined = combine_for_link_only(&ds_a, &ds_b);
    let lf = combined.lazy();

    let settings = Settings::builder(LinkType::LinkOnly)
        .comparison(common::fuzzy_comparison("first_name", 0.85))
        .comparison(common::exact_match_comparison("surname"))
        .blocking_rule(BlockingRule::on(&["surname"]))
        .source_dataset_column("source_dataset")
        .build()
        .unwrap();

    let mut linker = Linker::new(settings).unwrap();

    // Estimate u
    linker.estimate_u_using_random_sampling(&lf, 500).unwrap();

    // EM training
    linker
        .estimate_parameters_using_em(&lf, &BlockingRule::on(&["surname"]))
        .unwrap();

    // Predict
    let predictions = linker.predict(&lf, None).unwrap().collect().unwrap();

    // Should have scored pairs
    assert!(
        predictions.height() > 0,
        "LinkOnly should produce candidate pairs"
    );

    // Should have match_weight and match_probability columns
    let col_names: Vec<&str> = predictions
        .get_column_names()
        .into_iter()
        .map(|s| s.as_str())
        .collect();
    assert!(col_names.contains(&"match_weight"));
    assert!(col_names.contains(&"match_probability"));
}

#[test]
fn test_link_only_known_matches_score_high() {
    let (ds_a, ds_b) = make_two_datasets();
    let combined = combine_for_link_only(&ds_a, &ds_b);
    let lf = combined.lazy();

    let settings = Settings::builder(LinkType::LinkOnly)
        .comparison(common::fuzzy_comparison("first_name", 0.85))
        .comparison(common::exact_match_comparison("surname"))
        .comparison(common::exact_match_comparison("city"))
        .blocking_rule(BlockingRule::on(&["surname"]))
        .source_dataset_column("source_dataset")
        .build()
        .unwrap();

    let mut linker = Linker::new(settings).unwrap();
    linker.estimate_u_using_random_sampling(&lf, 500).unwrap();
    linker
        .estimate_parameters_using_em(&lf, &BlockingRule::on(&["surname"]))
        .unwrap();

    let predictions = linker.predict(&lf, None).unwrap().collect().unwrap();

    let uid_l = predictions.column("unique_id_l").unwrap().i64().unwrap();
    let uid_r = predictions.column("unique_id_r").unwrap().i64().unwrap();
    let probs = predictions
        .column("match_probability")
        .unwrap()
        .f64()
        .unwrap();

    // Known cross-dataset matches: (1,101) John/Jon Smith, (4,104) Alice/Alice Brown
    let mut found_cross_match = false;
    for ((l, r), p) in uid_l
        .into_iter()
        .zip(uid_r.into_iter())
        .zip(probs.into_iter())
    {
        if let (Some(l), Some(r), Some(prob)) = (l, r, p) {
            let is_cross_match = matches!(
                (l.min(r), l.max(r)),
                (1, 101) | (4, 104) | (3, 103) | (2, 102)
            );
            if is_cross_match && prob > 0.2 {
                found_cross_match = true;
            }
        }
    }
    assert!(
        found_cross_match,
        "At least one cross-dataset match should score reasonably"
    );
}
