//! Configurable performance benchmark — times each pipeline stage.
//!
//! Run:
//!   cargo run --example scaling --release              # Default: 100K records
//!   cargo run --example scaling --release -- 1000000   # 1M records
//!   cargo run --example scaling --release -- 50000     # 50K records

mod common;

use std::time::Instant;

use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let n_records: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    println!("weldrs scaling benchmark: {n_records} records");
    println!("{}", "=".repeat(50));

    // ── Data generation ──────────────────────────────────────────────
    let t = Instant::now();
    let df = common::generate_person_dataset(n_records, 0.15, 42);
    let elapsed = t.elapsed();
    println!(
        "Data generation:     {:.2}s  ({} rows)",
        elapsed.as_secs_f64(),
        df.height()
    );

    let lf = df.clone().lazy();

    // ── Build settings ───────────────────────────────────────────────
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
        .blocking_rule(BlockingRule::on(&["city"]))
        .build()?;

    let mut linker = Linker::new(settings)?;

    // ── Estimate lambda ──────────────────────────────────────────────
    let t = Instant::now();
    linker.estimate_probability_two_random_records_match(
        &lf,
        &[BlockingRule::on(&["first_name", "last_name"])],
        1.0,
    )?;
    println!("Estimate lambda:     {:.2}s", t.elapsed().as_secs_f64());

    // ── Estimate u ───────────────────────────────────────────────────
    let t = Instant::now();
    linker.estimate_u_using_random_sampling(&lf, 1_000)?;
    println!("Estimate u:          {:.2}s", t.elapsed().as_secs_f64());

    // ── EM pass 1: block on last_name ──────────────────────────────────
    let t = Instant::now();
    linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["last_name"]))?;
    println!("EM (last_name):        {:.2}s", t.elapsed().as_secs_f64());

    // ── EM pass 2: block on city ─────────────────────────────────────
    let t = Instant::now();
    linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["city"]))?;
    println!("EM (city):           {:.2}s", t.elapsed().as_secs_f64());

    // ── Predict ──────────────────────────────────────────────────────
    let t = Instant::now();
    let predictions = linker.predict(&lf, Some(0.0))?.collect()?;
    let n_pairs = predictions.height();
    println!(
        "Predict:             {:.2}s  ({n_pairs} pairs)",
        t.elapsed().as_secs_f64()
    );

    // ── Cluster ──────────────────────────────────────────────────────
    let t = Instant::now();
    let clusters = linker.cluster_pairwise_predictions(&predictions, 0.5)?;
    let n_clustered = clusters.height();
    println!(
        "Cluster:             {:.2}s  ({n_clustered} records)",
        t.elapsed().as_secs_f64()
    );

    Ok(())
}
