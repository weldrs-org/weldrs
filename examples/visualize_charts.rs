//! Chart visualization demo — generates SVG files for waterfall, match
//! weights, and weight distribution charts.
//!
//! Run: `cargo run --example visualize_charts --features visualize`

mod common;

use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::prelude::*;
use weldrs::visualize;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== weldrs Visualization Demo ===\n");

    // ── Dataset ──────────────────────────────────────────────────────
    let df = common::generate_person_dataset(50, 0.4, 42);
    println!("Generated {} rows.\n", df.height());

    // ── Settings ─────────────────────────────────────────────────────
    let settings = Settings::builder(LinkType::DedupeOnly)
        .comparison(
            ComparisonBuilder::new("first_name")
                .null_level()
                .exact_match_level()
                .jaro_winkler_level(0.88)
                .else_level()
                .build(),
        )
        .comparison(
            ComparisonBuilder::new("surname")
                .null_level()
                .exact_match_level()
                .else_level()
                .build(),
        )
        .comparison(
            ComparisonBuilder::new("city")
                .null_level()
                .exact_match_level()
                .levenshtein_level(2)
                .else_level()
                .build(),
        )
        .blocking_rule(BlockingRule::on(&["surname"]))
        .build()?;

    // ── Train ────────────────────────────────────────────────────────
    let mut linker = Linker::new(settings)?;
    let lf = df.lazy();

    linker.estimate_probability_two_random_records_match(
        &lf,
        &[BlockingRule::on(&["first_name", "surname"])],
        1.0,
    )?;
    linker.estimate_u_using_random_sampling(&lf, 200)?;
    linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["surname"]))?;
    linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["city"]))?;
    println!("Model trained.\n");

    // ── Predict ──────────────────────────────────────────────────────
    let predictions = linker.predict(&lf, None)?.collect()?;
    println!("Predicted {} pairs.\n", predictions.height());

    let opts = ChartOptions::default();

    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/charts");
    std::fs::create_dir_all(&out_dir)?;

    // ── 1. Waterfall chart ───────────────────────────────────────────
    let waterfall = linker.explain_pair(&predictions, 0)?;
    visualize::waterfall_chart_to_file(&waterfall, &out_dir.join("waterfall.svg"), &opts)?;
    println!("Wrote examples/charts/waterfall.svg");

    // ── 2. Match weights chart ───────────────────────────────────────
    let summary = linker.model_summary();
    visualize::match_weights_chart_to_file(&summary, &out_dir.join("match_weights.svg"), &opts)?;
    println!("Wrote examples/charts/match_weights.svg");

    // ── 3. Weight distribution histogram ─────────────────────────────
    let mw_col = predictions.column("match_weight")?.f64()?;
    let weights: Vec<f64> = mw_col.into_no_null_iter().collect();
    visualize::weight_distribution_chart_to_file(
        &weights,
        None,
        &out_dir.join("weight_distribution.svg"),
        &opts,
    )?;
    println!("Wrote examples/charts/weight_distribution.svg");

    println!("\nDone! Open the SVG files in a browser to view.");
    Ok(())
}
