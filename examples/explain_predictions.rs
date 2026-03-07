//! Waterfall pair explanation demo — shows why each pair got its score.
//!
//! Uses the same 10-row manual dataset as `basic_dedup.rs`. Trains a model,
//! predicts, then explains the top-scoring and bottom-scoring pairs with a
//! formatted waterfall display.
//!
//! Run: `cargo run --example explain_predictions`

use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::prelude::*;

fn fmt_weight(w: f64) -> String {
    if w.is_infinite() {
        format!("{:>12}", if w > 0.0 { "Inf" } else { "-Inf" })
    } else if w.is_nan() {
        format!("{:>12}", "NaN")
    } else {
        format!("{:>12.2}", w)
    }
}

fn print_waterfall(chart: &WaterfallChart) {
    println!(
        "\nWaterfall for pair ({}, {}):",
        chart.unique_id_l, chart.unique_id_r
    );
    println!(
        "{:<18}| {:<24}| {:<18}| {:>12} | {:>12}",
        "Step", "Level", "Values", "Match Weight", "Cumulative"
    );
    println!("{}", "-".repeat(92));

    for step in &chart.steps {
        let values = match (&step.value_l, &step.value_r) {
            (Some(l), Some(r)) => format!("{l} / {r}"),
            _ => String::new(),
        };
        println!(
            "{:<18}| {:<24}| {:<18}| {} | {}",
            step.column_name,
            step.label,
            values,
            fmt_weight(step.log2_bayes_factor),
            fmt_weight(step.cumulative_match_weight),
        );
    }

    let prob_str = if chart.final_match_probability.is_nan() {
        "~1.000".to_string()
    } else {
        format!("{:.3}", chart.final_match_probability)
    };
    let weight_str = if chart.final_match_weight.is_infinite() {
        "Inf".to_string()
    } else {
        format!("{:.2}", chart.final_match_weight)
    };
    println!("{:>68} Final: weight={}, prob={}", "", weight_str, prob_str,);
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== weldrs Waterfall Explanation Demo ===\n");

    // ── Step 1: Create a small dataset with known duplicates ─────────
    let df = df!(
        "unique_id" => [1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "first_name" => ["John", "Jane", "Bob", "Alice", "Eve",
                         "Jon", "Janet", "Robert", "Charlie", "Jane"],
        "last_name" => ["Smith", "Doe", "Williams", "Brown", "Davis",
                      "Smith", "Doe", "Williams", "Wilson", "Doe"],
        "city" => ["London", "Manchester", "Bristol", "Leeds", "York",
                   "London", "Manchester", "Bristol", "Oxford", "Manchster"],
        "email" => [Some("john@example.com"), Some("jane@example.com"), Some("bob@example.com"),
                    Some("alice@example.com"), None, Some("john@example.com"),
                    Some("janet@example.com"), Some("bob@example.com"), None,
                    Some("jane.doe@example.com")]
    )?;

    println!("Input data ({} rows):", df.height());
    println!("{df}\n");

    // ── Step 2: Configure comparisons and blocking ───────────────────
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
        .build()?;

    // ── Step 3: Train the model ──────────────────────────────────────
    let mut linker = Linker::new(settings)?;
    let lf = df.lazy();

    linker.estimate_probability_two_random_records_match(
        &lf,
        &[BlockingRule::on(&["first_name", "last_name"])],
        1.0,
    )?;
    linker.estimate_u_using_random_sampling(&lf, 200)?;
    linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["last_name"]))?;
    linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["city"]))?;
    println!("Model trained.\n");

    // ── Step 4: Predict (no threshold) ───────────────────────────────
    let predictions = linker.predict(&lf, None)?.collect()?;
    let sorted = predictions.sort(
        ["match_probability"],
        SortMultipleOptions::default().with_order_descending(true),
    )?;

    println!("Predicted {} pairs total.\n", sorted.height());

    // ── Step 5: Explain the highest-scoring pair ─────────────────────
    println!("--- Highest-scoring pair ---");
    let best = linker.explain_pair(&sorted, 0)?;
    print_waterfall(&best);

    // ── Step 6: Explain the lowest-scoring pair ──────────────────────
    let last_row = sorted.height() - 1;
    println!("\n--- Lowest-scoring pair ---");
    let worst = linker.explain_pair(&sorted, last_row)?;
    print_waterfall(&worst);

    // ── Step 7: JSON serialization of the waterfall ──────────────────
    let json = serde_json::to_string_pretty(&best)?;
    println!("\n--- Waterfall as JSON (first 500 chars) ---");
    println!("{}...", &json[..json.len().min(500)]);

    Ok(())
}
