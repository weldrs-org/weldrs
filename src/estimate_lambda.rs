//! Lambda estimation from deterministic rules.
//!
//! Lambda is the prior probability that two randomly chosen records are a
//! match. It is estimated by applying high-confidence blocking rules to
//! identify "certain" matches and dividing by the total number of possible
//! pairs, adjusted by an assumed recall.
//!
//! Most users call this via
//! [`Linker::estimate_probability_two_random_records_match`](crate::linker::Linker::estimate_probability_two_random_records_match)
//! rather than invoking the free function directly.
//!
//! See also: [`blocking`](crate::blocking) for how blocking rules are
//! constructed.

use polars::prelude::*;

use crate::blocking::BlockingRule;
use crate::error::{Result, WeldrsError};
use crate::settings::LinkType;

/// Estimate the probability that two random records match (lambda) using
/// deterministic rules.
///
/// Applies high-confidence blocking rules (e.g., exact match on multiple
/// columns) to identify "certain" matches, then divides by the total number
/// of possible pairs, adjusted by the estimated recall.
///
/// # Errors
///
/// Returns [`WeldrsError::Config`] if
/// `deterministic_rules` is empty. Returns
/// [`WeldrsError::Training`] if the
/// DataFrame has fewer than 2 records.
pub fn estimate_probability_two_random_records_match(
    lf: &LazyFrame,
    deterministic_rules: &[BlockingRule],
    link_type: &LinkType,
    unique_id_col: &str,
    source_dataset_column: Option<&str>,
    recall: f64,
) -> Result<f64> {
    if deterministic_rules.is_empty() {
        return Err(WeldrsError::Config(
            "At least one deterministic rule is required to estimate lambda".into(),
        ));
    }

    let collected = lf.clone().collect().map_err(|e| WeldrsError::Training {
        stage: "estimate_lambda",
        message: format!("Failed to collect: {e}"),
    })?;
    let n = collected.height() as f64;

    if n < 2.0 {
        return Err(WeldrsError::Training {
            stage: "estimate_lambda",
            message: "Need at least 2 records to estimate lambda".into(),
        });
    }

    // Total possible pairs.
    let total_pairs = match link_type {
        LinkType::DedupeOnly | LinkType::LinkAndDedupe => n * (n - 1.0) / 2.0,
        LinkType::LinkOnly => {
            let src_col = source_dataset_column.ok_or_else(|| {
                WeldrsError::Config(
                    "LinkOnly mode requires source_dataset_column to be set for lambda estimation"
                        .to_string(),
                )
            })?;
            // Count records per source dataset to compute n_a * n_b.
            let counts = collected
                .column(src_col)
                .map_err(|e| WeldrsError::Training {
                    stage: "estimate_lambda",
                    message: format!("Failed to access source column '{src_col}': {e}"),
                })?
                .as_materialized_series()
                .value_counts(false, false, PlSmallStr::from_static("count"), false)
                .map_err(|e| WeldrsError::Training {
                    stage: "estimate_lambda",
                    message: format!("Failed to count source datasets: {e}"),
                })?;
            let count_col = counts.column("count").map_err(|e| WeldrsError::Training {
                stage: "estimate_lambda",
                message: format!("Unexpected value_counts schema: {e}"),
            })?;
            let count_vals: Vec<f64> = count_col
                .cast(&DataType::Float64)
                .map_err(|e| WeldrsError::Training {
                    stage: "estimate_lambda",
                    message: format!("Failed to cast counts: {e}"),
                })?
                .f64()
                .map_err(|e| WeldrsError::Training {
                    stage: "estimate_lambda",
                    message: format!("Failed to read counts as f64: {e}"),
                })?
                .into_no_null_iter()
                .collect();
            if count_vals.len() != 2 {
                return Err(WeldrsError::Config(format!(
                    "LinkOnly expects exactly 2 source datasets, found {}",
                    count_vals.len()
                )));
            }
            count_vals[0] * count_vals[1]
        }
    };

    // Count matched pairs via deterministic rules.
    let uid_l = format!("{unique_id_col}_l");
    let uid_r = format!("{unique_id_col}_r");

    // Pre-select only the uid and blocking-key columns before joining.
    // This reduces the join payload from all columns to just the essentials.
    let needed_cols: std::collections::HashSet<&str> = {
        let mut s = std::collections::HashSet::new();
        s.insert(unique_id_col);
        if let Some(src_col) = source_dataset_column {
            s.insert(src_col);
        }
        for rule in deterministic_rules {
            for c in &rule.columns {
                s.insert(c.as_str());
            }
        }
        s
    };
    let select_exprs: Vec<Expr> = needed_cols.iter().map(|&c| col(c)).collect();

    let slim = collected.lazy().select(select_exprs);
    let left = slim.clone().select([col("*").name().suffix("_l")]);
    let right = slim.select([col("*").name().suffix("_r")]);

    let mut all_pairs: Vec<LazyFrame> = Vec::new();

    for rule in deterministic_rules {
        let left_on: Vec<Expr> = rule
            .columns
            .iter()
            .map(|c| col(format!("{c}_l").as_str()))
            .collect();
        let right_on: Vec<Expr> = rule
            .columns
            .iter()
            .map(|c| col(format!("{c}_r").as_str()))
            .collect();

        let joined = left.clone().join(
            right.clone(),
            left_on,
            right_on,
            JoinArgs::new(JoinType::Inner),
        );

        let filtered = match link_type {
            LinkType::DedupeOnly | LinkType::LinkAndDedupe => {
                joined.filter(col(uid_l.as_str()).lt(col(uid_r.as_str())))
            }
            LinkType::LinkOnly => {
                // source_dataset_column is guaranteed Some here (we error above otherwise).
                match source_dataset_column {
                    Some(src_col) => {
                        let src_l = format!("{src_col}_l");
                        let src_r = format!("{src_col}_r");
                        joined.filter(
                            col(src_l.as_str())
                                .neq(col(src_r.as_str()))
                                .and(col(uid_l.as_str()).lt(col(uid_r.as_str()))),
                        )
                    }
                    None => {
                        return Err(WeldrsError::Config(
                            "LinkOnly mode requires source_dataset_column".to_string(),
                        ));
                    }
                }
            }
        };

        all_pairs.push(filtered.select([col(uid_l.as_str()), col(uid_r.as_str())]));
    }

    let unioned = if all_pairs.len() == 1 {
        all_pairs.into_iter().next().unwrap()
    } else {
        concat(&all_pairs, UnionArgs::default()).map_err(|e| WeldrsError::Training {
            stage: "estimate_lambda",
            message: format!("Concat failed: {e}"),
        })?
    };

    let unique_pairs = unioned.unique(Some(cols([uid_l, uid_r])), UniqueKeepStrategy::First);

    let match_count = unique_pairs
        .collect()
        .map_err(|e| WeldrsError::Training {
            stage: "estimate_lambda",
            message: format!("Failed to count matches: {e}"),
        })?
        .height() as f64;

    let lambda = match_count / (total_pairs * recall);

    // Clamp to a reasonable range.
    Ok(lambda.clamp(1e-8, 0.99))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_lf() -> LazyFrame {
        use polars::prelude::*;
        df!(
            "unique_id" => [1i64, 2, 3, 4, 5, 6],
            "first_name" => ["John", "Jane", "Bob", "John", "Jane", "Alice"],
            "last_name" => ["Smith", "Doe", "Williams", "Smith", "Doe", "Brown"],
        )
        .unwrap()
        .lazy()
    }

    #[test]
    fn test_lambda_basic_estimate() {
        let lf = test_lf();
        // Block on first_name + last_name: finds exact duplicate pairs (1,4) and (2,5)
        let rules = vec![BlockingRule::on(&["first_name", "last_name"])];
        let lambda = estimate_probability_two_random_records_match(
            &lf,
            &rules,
            &LinkType::DedupeOnly,
            "unique_id",
            None,
            1.0, // recall = 1.0
        )
        .unwrap();

        // 6 records → C(6,2) = 15 total pairs. 2 matches found. lambda = 2/15 ≈ 0.133
        assert!(lambda > 0.0);
        assert!(lambda < 1.0);
        let expected = 2.0 / 15.0;
        assert!((lambda - expected).abs() < 1e-6);
    }

    #[test]
    fn test_lambda_clamped_low() {
        use polars::prelude::*;
        // All unique records → 0 matches → clamped to 1e-8
        let lf = df!(
            "unique_id" => [1i64, 2, 3],
            "name" => ["Alice", "Bob", "Carol"],
        )
        .unwrap()
        .lazy();

        let rules = vec![BlockingRule::on(&["name"])];
        let lambda = estimate_probability_two_random_records_match(
            &lf,
            &rules,
            &LinkType::DedupeOnly,
            "unique_id",
            None,
            1.0,
        )
        .unwrap();

        assert!((lambda - 1e-8).abs() < 1e-15);
    }

    #[test]
    fn test_lambda_errors_on_empty_rules() {
        let lf = test_lf();
        let result = estimate_probability_two_random_records_match(
            &lf,
            &[],
            &LinkType::DedupeOnly,
            "unique_id",
            None,
            1.0,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            WeldrsError::Config(_) => {}
            other => panic!("Expected Config error, got: {other:?}"),
        }
    }

    #[test]
    fn test_lambda_errors_on_tiny_df() {
        use polars::prelude::*;
        let lf = df!("unique_id" => [1i64], "name" => ["Alice"])
            .unwrap()
            .lazy();
        let rules = vec![BlockingRule::on(&["name"])];
        let result = estimate_probability_two_random_records_match(
            &lf,
            &rules,
            &LinkType::DedupeOnly,
            "unique_id",
            None,
            1.0,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            WeldrsError::Training { .. } => {}
            other => panic!("Expected Training error, got: {other:?}"),
        }
    }

    #[test]
    fn test_lambda_link_only_requires_source_column() {
        let lf = test_lf();
        let rules = vec![BlockingRule::on(&["first_name", "last_name"])];
        let result = estimate_probability_two_random_records_match(
            &lf,
            &rules,
            &LinkType::LinkOnly,
            "unique_id",
            None, // no source_dataset_column
            1.0,
        );
        assert!(
            result.is_err(),
            "LinkOnly without source column should error"
        );
    }

    #[test]
    fn test_lambda_link_only_with_source_column() {
        use polars::prelude::*;
        // Two "datasets" concatenated, distinguished by source_dataset
        let lf = df!(
            "unique_id" => [1i64, 2, 3, 101, 102, 103],
            "first_name" => ["John", "Jane", "Bob", "John", "Jane", "Alice"],
            "last_name" => ["Smith", "Doe", "Williams", "Smith", "Doe", "Brown"],
            "source_dataset" => ["a", "a", "a", "b", "b", "b"],
        )
        .unwrap()
        .lazy();

        let rules = vec![BlockingRule::on(&["first_name", "last_name"])];
        let lambda = estimate_probability_two_random_records_match(
            &lf,
            &rules,
            &LinkType::LinkOnly,
            "unique_id",
            Some("source_dataset"),
            1.0,
        )
        .unwrap();

        // n_a=3, n_b=3, total_pairs = 3*3 = 9
        // Matches: (John Smith, John Smith) and (Jane Doe, Jane Doe) = 2
        // lambda = 2 / 9 ≈ 0.222
        let expected = 2.0 / 9.0;
        assert!(
            (lambda - expected).abs() < 1e-6,
            "Expected lambda ≈ {expected}, got {lambda}"
        );
    }
}
