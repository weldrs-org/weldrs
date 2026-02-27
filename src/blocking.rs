//! Blocking rules and candidate-pair generation.
//!
//! Blocking restricts the comparison space by only generating record pairs
//! that agree on one or more "blocking" columns (an equi-join). Without
//! blocking every pair would be compared, which is O(n^2).
//!
//! This module covers **step 1** of the inference pipeline — producing
//! candidate pairs before [`comparison_vectors`](crate::comparison_vectors)
//! evaluates them and [`predict`](crate::predict) scores them.
//!
//! Blocking rules are also used during training: see
//! [`Settings::builder`](crate::settings::Settings::builder) for attaching
//! prediction-time rules and
//! [`Linker::estimate_parameters_using_em`](crate::linker::Linker::estimate_parameters_using_em)
//! for the training-time blocking rule.
//!
//! # Example
//!
//! ```
//! use weldrs::blocking::BlockingRule;
//!
//! // Block on surname — only pairs sharing a surname are compared.
//! let rule = BlockingRule::on(&["surname"]);
//!
//! // Block on city AND state (multi-column equi-join).
//! let strict = BlockingRule::on(&["city", "state"])
//!     .with_description("city + state block");
//! ```

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::settings::LinkType;

/// A blocking rule that defines which columns to equi-join on when generating
/// candidate record pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockingRule {
    /// Columns to equi-join on.
    pub columns: Vec<String>,
    /// Optional human-readable description of this blocking rule.
    pub description: Option<String>,
}

impl BlockingRule {
    /// Create a blocking rule that joins on the given columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use weldrs::blocking::BlockingRule;
    ///
    /// let rule = BlockingRule::on(&["surname"]);
    /// assert_eq!(rule.columns, vec!["surname"]);
    /// ```
    pub fn on(columns: &[&str]) -> Self {
        Self {
            columns: columns.iter().map(|s| s.to_string()).collect(),
            description: None,
        }
    }

    /// Attach a human-readable description to this blocking rule.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }
}

/// Suffix all columns in a DataFrame, adding `_l` or `_r`.
fn suffix_columns(df: &LazyFrame, suffix: &str) -> LazyFrame {
    df.clone().select([all().name().suffix(suffix)])
}

/// Generate candidate record pairs by applying blocking rules via equi-joins.
///
/// Returns a `LazyFrame` with columns suffixed `_l` and `_r`, plus a
/// `match_key` column indicating which blocking rule produced each pair.
///
/// # Errors
///
/// Returns an error if a Polars join or schema operation fails.
pub fn generate_blocked_pairs(
    df: &LazyFrame,
    blocking_rules: &[BlockingRule],
    link_type: &LinkType,
    unique_id_col: &str,
) -> Result<LazyFrame> {
    let uid_l = format!("{unique_id_col}_l");
    let uid_r = format!("{unique_id_col}_r");

    let mut left = suffix_columns(df, "_l");
    let mut right = suffix_columns(df, "_r");

    // Build a consistent column selection order for all blocking rules.
    let left_schema = left
        .collect_schema()
        .map_err(crate::error::WeldrsError::Polars)?;
    let right_schema = right
        .collect_schema()
        .map_err(crate::error::WeldrsError::Polars)?;
    let mut output_cols: Vec<Expr> = Vec::new();
    for name in left_schema.iter_names() {
        output_cols.push(col(name.as_str()));
    }
    for name in right_schema.iter_names() {
        output_cols.push(col(name.as_str()));
    }
    output_cols.push(col("match_key"));

    let mut all_pairs: Vec<LazyFrame> = Vec::new();

    for (i, rule) in blocking_rules.iter().enumerate() {
        // Build the join condition: equi-join on each blocking column.
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

        let mut joined = left.clone().join(
            right.clone(),
            left_on,
            right_on,
            JoinArgs::new(JoinType::Inner),
        );

        // Inner join drops the right key columns; re-add them from the left keys
        // (values are guaranteed equal by the join condition).
        for c in &rule.columns {
            joined =
                joined.with_column(col(format!("{c}_l").as_str()).alias(format!("{c}_r").as_str()));
        }

        // Filter out self-pairs and, for deduplication, keep only uid_l < uid_r.
        let filtered = match link_type {
            LinkType::DedupeOnly => joined.filter(col(uid_l.as_str()).lt(col(uid_r.as_str()))),
            LinkType::LinkOnly => {
                // For link-only the source dataset column must differ.
                // If no source column exists, use uid inequality as a fallback.
                joined.filter(col(uid_l.as_str()).neq(col(uid_r.as_str())))
            }
            LinkType::LinkAndDedupe => joined.filter(col(uid_l.as_str()).lt(col(uid_r.as_str()))),
        };

        let with_key = filtered
            .with_column(lit(i as u32).alias("match_key"))
            .select(output_cols.clone());
        all_pairs.push(with_key);
    }

    if all_pairs.is_empty() {
        return Ok(left
            .cross_join(right, None)
            .filter(col(uid_l.as_str()).lt(col(uid_r.as_str())))
            .with_column(lit(0u32).alias("match_key")));
    }

    // Incremental deduplication via anti-join: each subsequent rule's pairs
    // are anti-joined against the accumulated result before being appended.
    // This avoids a potentially expensive final `unique()` over the full union.
    let mut accumulated = all_pairs.remove(0);
    for extra in all_pairs {
        let new_only = extra.join(
            accumulated
                .clone()
                .select([col(uid_l.as_str()), col(uid_r.as_str())]),
            [col(uid_l.as_str()), col(uid_r.as_str())],
            [col(uid_l.as_str()), col(uid_r.as_str())],
            JoinArgs::new(JoinType::Anti),
        );
        accumulated = concat(&[accumulated, new_only], UnionArgs::default())?;
    }

    Ok(accumulated)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_df() -> LazyFrame {
        df!(
            "unique_id" => [1i64, 2, 3, 4],
            "first_name" => ["Alice", "Bob", "Alice", "Carol"],
            "city" => ["London", "London", "Paris", "London"],
        )
        .unwrap()
        .lazy()
    }

    #[test]
    fn test_dedupe_only_pairs() {
        let df = small_df();
        let rules = vec![BlockingRule::on(&["city"])];
        let pairs = generate_blocked_pairs(&df, &rules, &LinkType::DedupeOnly, "unique_id")
            .unwrap()
            .collect()
            .unwrap();

        let uid_l = pairs.column("unique_id_l").unwrap().i64().unwrap();
        let uid_r = pairs.column("unique_id_r").unwrap().i64().unwrap();

        for (l, r) in uid_l.into_iter().zip(uid_r.into_iter()) {
            assert!(l.unwrap() < r.unwrap(), "Expected uid_l < uid_r");
        }
    }

    #[test]
    fn test_cross_join_fallback() {
        let df = small_df();
        // Empty rules → cross-join fallback
        let pairs = generate_blocked_pairs(&df, &[], &LinkType::DedupeOnly, "unique_id")
            .unwrap()
            .collect()
            .unwrap();

        // 4 records → C(4,2) = 6 pairs
        assert_eq!(pairs.height(), 6);
    }

    #[test]
    fn test_multi_rule_deduplication() {
        let df = small_df();
        // Two rules that can produce overlapping pairs (city=London gives (1,2),(1,4),(2,4))
        let rules = vec![
            BlockingRule::on(&["city"]),
            BlockingRule::on(&["first_name"]),
        ];
        let pairs = generate_blocked_pairs(&df, &rules, &LinkType::DedupeOnly, "unique_id")
            .unwrap()
            .collect()
            .unwrap();

        // Check no duplicate (uid_l, uid_r) pairs
        let n_before = pairs.height();
        let deduped = pairs
            .lazy()
            .unique(
                Some(vec!["unique_id_l".to_string(), "unique_id_r".to_string()]),
                UniqueKeepStrategy::First,
            )
            .collect()
            .unwrap();
        assert_eq!(n_before, deduped.height());
    }

    #[test]
    fn test_match_key_assignment() {
        let df = small_df();
        let rules = vec![BlockingRule::on(&["city"])];
        let pairs = generate_blocked_pairs(&df, &rules, &LinkType::DedupeOnly, "unique_id")
            .unwrap()
            .collect()
            .unwrap();

        // All pairs from rule 0 should have match_key = 0
        let match_keys = pairs.column("match_key").unwrap();
        let cast = match_keys.cast(&DataType::UInt32).unwrap();
        for mk in cast.u32().unwrap().into_iter() {
            assert_eq!(mk, Some(0));
        }
    }

    #[test]
    fn test_suffixed_columns() {
        let df = small_df();
        let rules = vec![BlockingRule::on(&["city"])];
        let pairs = generate_blocked_pairs(&df, &rules, &LinkType::DedupeOnly, "unique_id")
            .unwrap()
            .collect()
            .unwrap();

        let col_names: Vec<&str> = pairs
            .get_column_names()
            .into_iter()
            .map(|s| s.as_str())
            .collect();
        assert!(col_names.contains(&"unique_id_l"));
        assert!(col_names.contains(&"unique_id_r"));
        assert!(col_names.contains(&"first_name_l"));
        assert!(col_names.contains(&"first_name_r"));
        assert!(col_names.contains(&"city_l"));
        assert!(col_names.contains(&"city_r"));
    }
}
