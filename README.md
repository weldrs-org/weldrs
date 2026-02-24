# weldrs

*Pronounced "welders"*

Fellegi-Sunter probabilistic record linkage in Rust, powered by [Polars](https://pola.rs/).

A Rust-native implementation inspired by the [Splink](https://github.com/moj-analytical-services/splink) Python project.

## Features

- **Blocking rules** — reduce the comparison space with equi-join blocking on one or more columns
- **Exact and fuzzy comparisons** — Jaro-Winkler, Levenshtein, and Jaro similarity predicates alongside exact matching
- **EM training** — unsupervised Expectation-Maximisation to learn m/u probabilities
- **Fellegi-Sunter scoring** — Bayes-factor match weights and match probabilities for every candidate pair
- **Connected-components clustering** — union-find grouping of linked records
- **Model serialization** — save and load trained model parameters as JSON
- **Waterfall explanations** — step-by-step breakdowns showing why each pair received its score

## Quick Start

```rust
use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::prelude::*;

fn main() -> Result<()> {
    // 1. Prepare your data as a Polars DataFrame
    let df = df!(
        "unique_id"  => [1i64, 2, 3, 4],
        "first_name" => ["John", "Jane", "Jon", "Jane"],
        "surname"    => ["Smith", "Doe", "Smith", "Doe"],
    )?;

    // 2. Define comparisons and build settings
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
        .blocking_rule(BlockingRule::on(&["surname"]))
        .build()?;

    // 3. Train the model
    let mut linker = Linker::new(settings)?;
    let lf = df.lazy();

    linker.estimate_probability_two_random_records_match(
        &lf,
        &[BlockingRule::on(&["first_name", "surname"])],
        1.0,
    )?;
    linker.estimate_u_using_random_sampling(&lf, 200)?;
    linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["surname"]))?;

    // 4. Predict — score all candidate pairs
    let predictions = linker.predict(&lf, None)?.collect()?;

    // 5. Cluster — group linked records
    let clusters = linker.cluster_pairwise_predictions(&predictions, 0.5)?;
    println!("{clusters}");

    Ok(())
}
```

## Concepts

**weldrs** implements the Fellegi-Sunter model of probabilistic record linkage:

- **Comparisons** define how record pairs are evaluated. Each comparison targets one input column (e.g. `first_name`) and contains multiple **levels** ordered from most to least specific (e.g. exact match, Jaro-Winkler >= 0.88, else).

- **m-probability** — the probability that a comparison level agrees, given the records *are* a true match.

- **u-probability** — the probability that a comparison level agrees, given the records *are not* a match.

- **Bayes factor** — the ratio m/u. A Bayes factor > 1 provides evidence towards a match; < 1 provides evidence against.

- **Match weight** — log2 of the Bayes factor. The final match weight for a pair is the sum of the prior (from lambda) plus each comparison's individual match weight.

- **Lambda** — the prior probability that two randomly chosen records are a match. Estimated from deterministic rules or set manually.

- **Blocking rules** — equi-join predicates that restrict which record pairs are compared, making linkage tractable on large datasets.

## Detailed Usage Guide

### Defining comparisons

Use `ComparisonBuilder` to define how columns are compared. Levels are evaluated top-to-bottom; the first matching level wins.

```rust
use weldrs::comparison::ComparisonBuilder;

let name_comparison = ComparisonBuilder::new("first_name")
    .null_level()                    // both values null
    .exact_match_level()             // exact string equality
    .jaro_winkler_level(0.88)        // fuzzy: Jaro-Winkler >= 0.88
    .else_level()                    // everything else
    .build();

let city_comparison = ComparisonBuilder::new("city")
    .null_level()
    .exact_match_level()
    .levenshtein_level(2)            // fuzzy: edit distance <= 2
    .else_level()
    .build();
```

Available fuzzy predicates:
- `jaro_winkler_level(threshold)` — Jaro-Winkler similarity >= threshold
- `jaro_level(threshold)` — Jaro similarity >= threshold
- `levenshtein_level(threshold)` — Levenshtein edit distance <= threshold

### Configuring settings

```rust
use weldrs::prelude::*;

let settings = Settings::builder(LinkType::DedupeOnly)
    .comparison(name_comparison)
    .comparison(city_comparison)
    .blocking_rule(BlockingRule::on(&["surname"]))
    .blocking_rule(BlockingRule::on(&["city"]))
    .unique_id_column("record_id")                    // default: "unique_id"
    .probability_two_random_records_match(0.001)      // default: 0.0001
    .build()?;
```

`LinkType` options:
- `DedupeOnly` — find duplicates within a single dataset
- `LinkOnly` — link records between two datasets
- `LinkAndDedupe` — link and deduplicate combined

### Blocking rules

Blocking rules define equi-join conditions to reduce the comparison space. Without blocking, every pair of records would be compared (quadratic).

```rust
use weldrs::prelude::*;

// Block on a single column
let rule = BlockingRule::on(&["surname"]);

// Block on multiple columns (AND condition)
let strict_rule = BlockingRule::on(&["first_name", "surname"]);

// Add an optional description
let rule = BlockingRule::on(&["city"]).with_description("Same city");
```

### Training

Training estimates the model parameters in three steps:

```rust
// 1. Estimate lambda (prior match probability) from deterministic rules
linker.estimate_probability_two_random_records_match(
    &lf,
    &[BlockingRule::on(&["first_name", "surname"])],
    1.0,  // assumed recall
)?;

// 2. Estimate u-probabilities from random record pairs
linker.estimate_u_using_random_sampling(&lf, 1_000)?;

// 3. EM training passes — each pass fixes comparisons that overlap
//    with the blocking rule and trains the rest
linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["surname"]))?;
linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["city"]))?;
```

### Prediction

```rust
// Score all candidate pairs (no threshold)
let predictions = linker.predict(&lf, None)?.collect()?;

// Score with a minimum match-weight threshold
let predictions = linker.predict(&lf, Some(0.0))?.collect()?;
```

The resulting DataFrame includes `match_weight`, `match_probability`, and individual `bf_*` columns for each comparison.

### Clustering

```rust
// Group linked records into clusters using a match-probability threshold
let clusters = linker.cluster_pairwise_predictions(&predictions, 0.5)?;
// Returns a DataFrame with columns: [unique_id, cluster_id]
```

### Explaining predictions

Waterfall charts show exactly how each comparison contributed to a pair's score:

```rust
// Explain a single pair (by row index in the predictions DataFrame)
let chart = linker.explain_pair(&predictions, 0)?;

for step in &chart.steps {
    println!("{}: {} (weight: {:.2})", step.column_name, step.label, step.log2_bayes_factor);
}

// Explain multiple pairs at once
let charts = linker.explain_pairs(&predictions, &[0, 1, 2])?;
```

### Inspecting the model

```rust
let summary = linker.model_summary();
println!("Lambda: {:.6}", summary.probability_two_random_records_match);

for comp in &summary.comparisons {
    println!("Comparison: {}", comp.output_column_name);
    for level in &comp.levels {
        println!("  {} — BF: {:?}, weight: {:?}",
            level.label, level.bayes_factor, level.log2_bayes_factor);
    }
}
```

### Saving and loading trained models

```rust
// Save trained model to JSON
let json = linker.save_settings_json()?;
std::fs::write("model.json", &json)?;

// Load a previously trained model
let json = std::fs::read_to_string("model.json")?;
let restored_linker = Linker::load_settings_json(&json)?;

// Use the restored linker for prediction — no retraining needed
let predictions = restored_linker.predict(&lf, None)?.collect()?;
```

## Examples

Run any example with `cargo run --example <name>`:

| Example | Description |
|---|---|
| `cargo run --example basic_dedup` | Full pipeline tutorial with a 10-row dataset |
| `cargo run --example fuzzy_matching` | Compare exact-only vs. fuzzy comparisons on 1K rows |
| `cargo run --example explain_predictions` | Waterfall explanation of top/bottom scoring pairs |
| `cargo run --example model_parameters` | Inspect trained m/u/BF parameters |
| `cargo run --example save_and_load` | Serialize and restore a trained model |
| `cargo run --example scaling --release` | Performance benchmark (default 100K rows) |

## License

MIT
