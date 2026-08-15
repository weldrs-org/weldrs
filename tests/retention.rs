use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::prelude::*;

fn data() -> DataFrame {
    df!(
        "unique_id" => [1i64, 2, 3, 4],
        "first_name" => ["John", "John", "Jane", "Jane"],
        "city" => ["London", "London", "Paris", "Paris"],
        "notes" => ["a", "b", "c", "d"],
    )
    .unwrap()
}

fn build(retain_matching: bool, retain_intermediate: bool, extra: &[&str]) -> Linker {
    let settings = Settings::builder(LinkType::DedupeOnly)
        .comparison(
            ComparisonBuilder::new("first_name")
                .null_level()
                .exact_match_level()
                .else_level()
                .build()
                .unwrap(),
        )
        .blocking_rule(BlockingRule::on(&["city"]))
        .retain_matching_columns(retain_matching)
        .retain_intermediate_calculation_columns(retain_intermediate)
        .additional_columns_to_retain(extra)
        .build()
        .unwrap();
    Linker::new(settings).unwrap()
}

fn columns(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

#[test]
fn test_default_retention_drops_intermediate_keeps_matching() {
    let linker = build(true, false, &[]);
    let preds = linker
        .predict(&data().lazy(), None)
        .unwrap()
        .collect()
        .unwrap();
    let cols = columns(&preds);

    // Always present.
    for c in [
        "unique_id_l",
        "unique_id_r",
        "match_weight",
        "match_probability",
        "gamma_first_name",
    ] {
        assert!(cols.iter().any(|x| x == c), "expected {c} in {cols:?}");
    }
    // Matching columns kept (default).
    assert!(cols.iter().any(|x| x == "first_name_l"));
    // Intermediate bf_ dropped (default).
    assert!(
        !cols.iter().any(|x| x == "bf_first_name"),
        "bf_ should be dropped by default"
    );
}

#[test]
fn test_retain_intermediate_keeps_bf() {
    let linker = build(true, true, &[]);
    let preds = linker
        .predict(&data().lazy(), None)
        .unwrap()
        .collect()
        .unwrap();
    assert!(columns(&preds).iter().any(|x| x == "bf_first_name"));
}

#[test]
fn test_drop_matching_columns() {
    let linker = build(false, false, &[]);
    let preds = linker
        .predict(&data().lazy(), None)
        .unwrap()
        .collect()
        .unwrap();
    let cols = columns(&preds);
    assert!(
        !cols.iter().any(|x| x == "first_name_l"),
        "matching cols should be dropped"
    );
    // Gamma still kept.
    assert!(cols.iter().any(|x| x == "gamma_first_name"));
}

#[test]
fn test_additional_columns_to_retain() {
    let linker = build(false, false, &["notes"]);
    let preds = linker
        .predict(&data().lazy(), None)
        .unwrap()
        .collect()
        .unwrap();
    let cols = columns(&preds);
    assert!(cols.iter().any(|x| x == "notes_l"));
    assert!(cols.iter().any(|x| x == "notes_r"));
}
