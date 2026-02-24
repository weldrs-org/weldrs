//! Basic deduplication tutorial — 10 rows, every step of the pipeline.
//!
//! Run: `cargo run --example basic_dedup`

use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // ── Step 1: Create a small dataset with known duplicates ─────────
    //
    // Three duplicate clusters:
    //   - Cluster A: ids 1, 6  — John/Jon Smith, London
    //   - Cluster B: ids 2, 7, 10 — Jane/Janet/Jane Doe, Manchester/Manchster
    //   - Cluster C: ids 3, 8  — Bob/Robert Williams, Bristol
    // Singletons: 4 (Alice Brown), 5 (Eve Davis), 9 (Charlie Wilson)
    let df = df!(
        "unique_id" => [1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "first_name" => ["John", "Jane", "Bob", "Alice", "Eve",
                         "Jon", "Janet", "Robert", "Charlie", "Jane"],
        "surname" => ["Smith", "Doe", "Williams", "Brown", "Davis",
                      "Smith", "Doe", "Williams", "Wilson", "Doe"],
        "city" => ["London", "Manchester", "Bristol", "Leeds", "York",
                   "London", "Manchester", "Bristol", "Oxford", "Manchster"],
        "email" => [Some("john@example.com"), Some("jane@example.com"), Some("bob@example.com"),
                    Some("alice@example.com"), None, Some("john@example.com"),
                    Some("janet@example.com"), Some("bob@example.com"), None,
                    Some("jane.doe@example.com")]
    )?;

    println!("=== weldrs Basic Deduplication Tutorial ===\n");
    println!("Input data ({} rows):", df.height());
    println!("{df}\n");

    // ── Step 2: Configure comparisons and blocking ───────────────────
    //
    // We define three comparisons:
    //   - first_name: null check → exact match → Jaro-Winkler ≥ 0.88 → else
    //   - surname:    null check → exact match → else
    //   - city:       null check → exact match → Levenshtein ≤ 2 → else
    //
    // Blocking rule: generate candidate pairs where surname matches exactly.
    // This dramatically reduces the number of comparisons needed.
    let first_name_comparison = ComparisonBuilder::new("first_name")
        .null_level()
        .exact_match_level()
        .jaro_winkler_level(0.88)
        .else_level()
        .build();

    let surname_comparison = ComparisonBuilder::new("surname")
        .null_level()
        .exact_match_level()
        .else_level()
        .build();

    let city_comparison = ComparisonBuilder::new("city")
        .null_level()
        .exact_match_level()
        .levenshtein_level(2)
        .else_level()
        .build();

    let settings = Settings::builder(LinkType::DedupeOnly)
        .comparison(first_name_comparison)
        .comparison(surname_comparison)
        .comparison(city_comparison)
        .blocking_rule(BlockingRule::on(&["surname"]))
        .build()?;

    println!(
        "Configured {} comparisons, {} blocking rule(s)\n",
        settings.comparisons.len(),
        settings.blocking_rules.len(),
    );

    // ── Step 3: Create a Linker and train the model ──────────────────
    let mut linker = Linker::new(settings)?;
    let lf = df.clone().lazy();

    // Estimate lambda: the probability that two random records are a match.
    // Uses a deterministic rule (exact match on first_name AND surname) with
    // an assumed recall of 1.0.
    let lambda = linker.estimate_probability_two_random_records_match(
        &lf,
        &[BlockingRule::on(&["first_name", "surname"])],
        1.0,
    )?;
    println!("Estimated lambda (P(random pair is match)): {lambda:.6}");

    // Estimate u-probabilities from random record pairs.
    // u = probability of agreement given the records are NOT a match.
    linker.estimate_u_using_random_sampling(&lf, 200)?;
    println!("Estimated u-probabilities from random sampling");

    // EM pass 1: block on surname — trains m/u for first_name and city
    // (surname is fixed since it's always equal under this blocking rule).
    linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["surname"]))?;
    println!("EM pass 1 complete (blocked on surname)");

    // EM pass 2: block on city — trains m/u for first_name and surname
    // (city is fixed since it's always equal under this blocking rule).
    linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["city"]))?;
    println!("EM pass 2 complete (blocked on city)\n");

    // ── Step 4: Predict — score all candidate pairs ──────────────────
    let predictions = linker.predict(&lf, None)?.collect()?;

    // Sort by match_probability descending for display
    let sorted = predictions.sort(
        ["match_probability"],
        SortMultipleOptions::default().with_order_descending(true),
    )?;

    println!("Predicted pairs ({} total):", sorted.height());
    println!(
        "{}",
        sorted.select([
            "unique_id_l",
            "unique_id_r",
            "first_name_l",
            "first_name_r",
            "surname_l",
            "match_weight",
            "match_probability",
        ])?
    );

    // ── Step 5: Cluster — group linked records ───────────────────────
    let clusters = linker.cluster_pairwise_predictions(&predictions, 0.5)?;

    let sorted_clusters =
        clusters.sort(["cluster_id", "unique_id"], SortMultipleOptions::default())?;

    println!("\nClusters (threshold: 0.5 match probability):");
    println!("{sorted_clusters}");

    Ok(())
}
