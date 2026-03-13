mod common;

use polars::prelude::*;

#[test]
fn test_waterfall_from_trained_linker() {
    let df = common::make_test_df();
    let linker = common::make_trained_linker(&df);

    let predictions = linker
        .predict(&df.clone().lazy(), None)
        .unwrap()
        .collect()
        .unwrap();

    assert!(predictions.height() > 0);

    // Explain the first row
    let chart = linker.explain_pair(&predictions, 0).unwrap();

    // Prior + 3 comparisons (first_name, surname, city) = 4 steps
    assert_eq!(chart.steps.len(), 4);

    // First step is the prior
    assert_eq!(chart.steps[0].column_name, "Prior");
    assert_eq!(chart.steps[0].label, "Prior (lambda)");
    assert!(chart.steps[0].comparison_vector_value.is_none());

    // Remaining steps are comparisons
    assert_eq!(chart.steps[1].column_name, "first_name");
    assert_eq!(chart.steps[2].column_name, "surname");
    assert_eq!(chart.steps[3].column_name, "city");

    // Each comparison step has a gamma value
    for step in &chart.steps[1..] {
        assert!(step.comparison_vector_value.is_some());
    }

    // Cumulative progression is correct
    let mut running = 0.0;
    for step in &chart.steps {
        running += step.log2_bayes_factor;
        assert!(
            (step.cumulative_match_weight - running).abs() < 1e-10,
            "Cumulative mismatch at step '{}'",
            step.column_name
        );
    }

    // Final values come from the DataFrame
    let df_weight = predictions
        .column("match_weight")
        .unwrap()
        .f64()
        .unwrap()
        .get(0)
        .unwrap();
    let df_prob = predictions
        .column("match_probability")
        .unwrap()
        .f64()
        .unwrap()
        .get(0)
        .unwrap();
    assert!(
        (chart.final_match_weight - df_weight).abs() < 1e-6,
        "final_match_weight {} != DataFrame {}",
        chart.final_match_weight,
        df_weight
    );
    assert!(
        (chart.final_match_probability - df_prob).abs() < 1e-6,
        "final_match_probability {} != DataFrame {}",
        chart.final_match_probability,
        df_prob
    );

    // Serializes to JSON and back
    let json = serde_json::to_string(&chart).unwrap();
    let restored: weldrs::prelude::WaterfallChart = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.steps.len(), chart.steps.len());
    assert_eq!(restored.unique_id_l, chart.unique_id_l);
}

#[test]
fn test_model_summary_from_trained_linker() {
    let df = common::make_test_df();
    let linker = common::make_trained_linker(&df);

    let summary = linker.model_summary();

    // 3 comparisons: first_name, surname, city
    assert_eq!(summary.comparisons.len(), 3);
    assert_eq!(summary.comparisons[0].output_column_name, "first_name");
    assert_eq!(summary.comparisons[1].output_column_name, "surname");
    assert_eq!(summary.comparisons[2].output_column_name, "city");

    // Prior matches settings
    assert!(
        (summary.probability_two_random_records_match
            - linker.settings().probability_two_random_records_match)
            .abs()
            < 1e-10
    );

    // Each comparison has levels with the null level last
    for comp in &summary.comparisons {
        assert!(!comp.levels.is_empty());
        assert!(
            comp.levels.last().unwrap().is_null_level,
            "Null level should be last for '{}'",
            comp.output_column_name
        );
    }

    // JSON round-trip
    let json = serde_json::to_string(&summary).unwrap();
    let restored: weldrs::prelude::ModelSummary = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.comparisons.len(), summary.comparisons.len());
    assert!((restored.prior_match_weight - summary.prior_match_weight).abs() < 1e-10);
}
