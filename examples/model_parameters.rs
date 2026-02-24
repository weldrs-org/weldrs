//! Model parameter summary demo — inspect trained parameters.
//!
//! Generates a 5,000-record synthetic dataset, trains a model, then prints
//! a structured parameter summary showing the prior and each comparison's
//! level parameters.
//!
//! Run: `cargo run --example model_parameters`

mod common;

use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== weldrs Model Parameter Summary Demo ===\n");

    // ── Step 1: Generate a realistic synthetic dataset ────────────────
    let df = common::generate_person_dataset(5_000, 0.15, 42);
    println!("Generated dataset: {} rows\n", df.height());

    let lf = df.lazy();

    // ── Step 2: Configure comparisons and blocking ───────────────────
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
                .jaro_winkler_level(0.92)
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
        .blocking_rule(BlockingRule::on(&["city"]))
        .build()?;

    // ── Step 3: Train the model ──────────────────────────────────────
    let mut linker = Linker::new(settings)?;

    linker.estimate_probability_two_random_records_match(
        &lf,
        &[BlockingRule::on(&["first_name", "surname"])],
        1.0,
    )?;
    linker.estimate_u_using_random_sampling(&lf, 1_000)?;
    linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["surname"]))?;
    linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["city"]))?;
    println!("Model trained.\n");

    // ── Step 4: Print the model summary ──────────────────────────────
    let summary = linker.model_summary();

    println!("=== Prior ===");
    println!(
        "  Lambda (P(match)):  {:.6}",
        summary.probability_two_random_records_match
    );
    println!("  Prior Bayes factor: {:.4}", summary.prior_bayes_factor);
    println!("  Prior match weight: {:.2}\n", summary.prior_match_weight);

    for comp in &summary.comparisons {
        println!("Comparison: {}", comp.output_column_name);
        println!(
            "  {:<24}| {:>3} | {:<10} | {:<10} | {:<10} | {:>7}",
            "Level", "cv", "m", "u", "BF", "Weight"
        );
        println!("  {}", "-".repeat(76));

        for level in &comp.levels {
            if level.is_null_level {
                println!(
                    "  {:<24}| {:>3} | {:<10} | {:<10} | {:<10} | {:>7}",
                    level.label,
                    level.comparison_vector_value,
                    "\u{2014}",
                    "\u{2014}",
                    "\u{2014}",
                    "\u{2014}",
                );
            } else {
                println!(
                    "  {:<24}| {:>3} | {:<10.6} | {:<10.6} | {:<10.4} | {:>7.2}",
                    level.label,
                    level.comparison_vector_value,
                    level.m_probability.unwrap_or(0.0),
                    level.u_probability.unwrap_or(0.0),
                    level.bayes_factor.unwrap_or(0.0),
                    level.log2_bayes_factor.unwrap_or(0.0),
                );
            }
        }
        println!();
    }

    // ── Step 5: JSON preview ─────────────────────────────────────────
    let json = serde_json::to_string_pretty(&summary)?;
    println!("--- Model summary as JSON (first 500 chars) ---");
    println!("{}...", &json[..json.len().min(500)]);

    Ok(())
}
