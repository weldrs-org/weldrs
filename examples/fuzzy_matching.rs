//! Fuzzy matching demo — shows how fuzzy comparisons improve match quality.
//!
//! Runs two linkage passes on the same ~1K-row generated dataset:
//! one with exact-only comparisons, one with fuzzy (Jaro-Winkler/Levenshtein).
//!
//! Run: `cargo run --example fuzzy_matching`

mod common;

use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== weldrs Fuzzy Matching Demo ===\n");

    // Generate 1,000 unique records with a 15% duplication rate
    let df = common::generate_person_dataset(1_000, 0.15, 42);
    common::print_df_summary(&df, "Generated dataset");

    let lf = df.clone().lazy();
    let blocking_rules = [BlockingRule::on(&["surname"]), BlockingRule::on(&["city"])];

    // ── Run 1: Exact-only comparisons ────────────────────────────────
    println!("\n--- Run 1: Exact-only comparisons ---");

    let exact_settings = Settings::builder(LinkType::DedupeOnly)
        .comparison(
            ComparisonBuilder::new("first_name")
                .null_level()
                .exact_match_level()
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
                .else_level()
                .build(),
        )
        .blocking_rule(blocking_rules[0].clone())
        .blocking_rule(blocking_rules[1].clone())
        .build()?;

    let mut exact_linker = Linker::new(exact_settings)?;
    exact_linker.estimate_probability_two_random_records_match(
        &lf,
        &[BlockingRule::on(&["first_name", "surname"])],
        1.0,
    )?;
    exact_linker.estimate_u_using_random_sampling(&lf, 1_000)?;
    exact_linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["surname"]))?;
    exact_linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["city"]))?;

    let exact_predictions = exact_linker.predict(&lf, None)?.collect()?;
    let exact_high = exact_predictions
        .clone()
        .lazy()
        .filter(col("match_probability").gt(lit(0.5)))
        .collect()?
        .height();

    println!("Total pairs: {}", exact_predictions.height());
    println!("High-confidence pairs (probability > 0.5): {exact_high}");

    // ── Run 2: Fuzzy comparisons ─────────────────────────────────────
    println!("\n--- Run 2: Fuzzy comparisons ---");

    let fuzzy_settings = Settings::builder(LinkType::DedupeOnly)
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
        .blocking_rule(blocking_rules[0].clone())
        .blocking_rule(blocking_rules[1].clone())
        .build()?;

    let mut fuzzy_linker = Linker::new(fuzzy_settings)?;
    fuzzy_linker.estimate_probability_two_random_records_match(
        &lf,
        &[BlockingRule::on(&["first_name", "surname"])],
        1.0,
    )?;
    fuzzy_linker.estimate_u_using_random_sampling(&lf, 1_000)?;
    fuzzy_linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["surname"]))?;
    fuzzy_linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["city"]))?;

    let fuzzy_predictions = fuzzy_linker.predict(&lf, None)?.collect()?;
    let fuzzy_high = fuzzy_predictions
        .clone()
        .lazy()
        .filter(col("match_probability").gt(lit(0.5)))
        .collect()?
        .height();

    println!("Total pairs: {}", fuzzy_predictions.height());
    println!("High-confidence pairs (probability > 0.5): {fuzzy_high}");

    // ── Comparison ───────────────────────────────────────────────────
    let diff = fuzzy_high as i64 - exact_high as i64;
    println!("\n=== Results ===");
    println!("Exact-only found {exact_high} high-confidence pairs");
    println!("Fuzzy found {fuzzy_high} high-confidence pairs ({diff:+} additional matches)");

    Ok(())
}
