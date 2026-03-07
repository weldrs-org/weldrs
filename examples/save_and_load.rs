//! Serialization round-trip demo — train, save, load, and verify predictions match.
//!
//! Run: `cargo run --example save_and_load`

mod common;

use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== weldrs Save & Load Demo ===\n");

    // ── Step 1: Generate data and train ──────────────────────────────
    let df = common::generate_person_dataset(10_000, 0.15, 99);
    common::print_df_summary(&df, "Generated dataset");

    let lf = df.clone().lazy();

    let settings = Settings::builder(LinkType::DedupeOnly)
        .comparison(
            ComparisonBuilder::new("first_name")
                .null_level()
                .exact_match_level()
                .jaro_winkler_level(0.88)
                .else_level()
                .build()?,
        )
        .comparison(
            ComparisonBuilder::new("last_name")
                .null_level()
                .exact_match_level()
                .jaro_winkler_level(0.92)
                .else_level()
                .build()?,
        )
        .comparison(
            ComparisonBuilder::new("city")
                .null_level()
                .exact_match_level()
                .levenshtein_level(2)
                .else_level()
                .build()?,
        )
        .blocking_rule(BlockingRule::on(&["last_name"]))
        .blocking_rule(BlockingRule::on(&["city"]))
        .build()?;

    let mut linker = Linker::new(settings)?;

    linker.estimate_probability_two_random_records_match(
        &lf,
        &[BlockingRule::on(&["first_name", "last_name"])],
        1.0,
    )?;
    linker.estimate_u_using_random_sampling(&lf, 1_000)?;
    linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["last_name"]))?;
    linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["city"]))?;

    println!("\nModel trained successfully.");

    // ── Step 2: Predict with original linker ─────────────────────────
    let original_predictions = linker.predict(&lf, Some(0.0))?.collect()?;
    println!(
        "Original linker predictions: {} pairs",
        original_predictions.height()
    );

    // ── Step 3: Save to JSON ─────────────────────────────────────────
    let json = linker.save_settings_json()?;
    println!("\nSaved settings to JSON ({} bytes)", json.len());
    println!("Preview (first 500 chars):");
    println!("{}", &json[..json.len().min(500)]);
    println!("...\n");

    // ── Step 4: Load from JSON ───────────────────────────────────────
    let restored_linker = Linker::load_settings_json(&json)?;
    println!("Restored linker from JSON");

    // ── Step 5: Predict with restored linker ─────────────────────────
    let restored_predictions = restored_linker.predict(&lf, Some(0.0))?.collect()?;
    println!(
        "Restored linker predictions: {} pairs",
        restored_predictions.height()
    );

    // ── Step 6: Compare ──────────────────────────────────────────────
    assert_eq!(
        original_predictions.height(),
        restored_predictions.height(),
        "Prediction counts should match after round-trip"
    );
    println!("\nRound-trip verified: same number of prediction rows.");

    // ── Step 7: Print trained parameters ─────────────────────────────
    println!("\n--- Trained Parameters ---");
    for comp in &restored_linker.settings.comparisons {
        println!("\nComparison: {}", comp.output_column_name);
        for level in &comp.comparison_levels {
            if level.is_null_level {
                println!("  [null]  (neutral — Bayes factor = 1.0)");
                continue;
            }
            let m = level
                .m_probability
                .map_or("N/A".to_string(), |v| format!("{v:.6}"));
            let u = level
                .u_probability
                .map_or("N/A".to_string(), |v| format!("{v:.6}"));
            let bf = level
                .bayes_factor()
                .map_or("N/A".to_string(), |v| format!("{v:.4}"));
            println!(
                "  [cv={}] {:<30}  m={m}  u={u}  BF={bf}",
                level.comparison_vector_value, level.label,
            );
        }
    }

    Ok(())
}
