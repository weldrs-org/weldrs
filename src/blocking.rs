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
//! // Block on last_name — only pairs sharing a last_name are compared.
//! let rule = BlockingRule::on(&["last_name"]);
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
    /// Columns to equi-join on (empty for a custom-predicate rule).
    pub columns: Vec<String>,
    /// Optional human-readable description of this blocking rule.
    pub description: Option<String>,
    /// Optional custom boolean condition in Polars SQL over the suffixed
    /// `{col}_l` / `{col}_r` columns. When set, the rule is evaluated as a
    /// cross-join + filter instead of an equi-join (O(n²) — use sparingly).
    #[serde(default)]
    pub predicate_dsl: Option<String>,
}

impl BlockingRule {
    /// Create a blocking rule that equi-joins on the given columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use weldrs::blocking::BlockingRule;
    ///
    /// let rule = BlockingRule::on(&["last_name"]);
    /// assert_eq!(rule.columns, vec!["last_name"]);
    /// ```
    pub fn on(columns: &[&str]) -> Self {
        Self {
            columns: columns.iter().map(|s| s.to_string()).collect(),
            description: None,
            predicate_dsl: None,
        }
    }

    /// Create a blocking rule from a custom Polars-SQL boolean condition over
    /// the suffixed `{col}_l` / `{col}_r` columns (Splink's `CustomRule`
    /// analog). Evaluated as a cross-join + filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use weldrs::blocking::BlockingRule;
    ///
    /// let rule = BlockingRule::custom("substr(surname_l, 1, 1) = substr(surname_r, 1, 1)");
    /// ```
    pub fn custom(dsl: &str) -> Self {
        Self {
            columns: Vec::new(),
            description: None,
            predicate_dsl: Some(dsl.to_string()),
        }
    }

    /// Attach a human-readable description to this blocking rule.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// The boolean SQL condition this rule represents, over suffixed columns.
    fn condition_sql(&self) -> String {
        if let Some(dsl) = &self.predicate_dsl {
            dsl.clone()
        } else if self.columns.is_empty() {
            "true".to_string()
        } else {
            self.columns
                .iter()
                .map(|c| format!("{c}_l = {c}_r"))
                .collect::<Vec<_>>()
                .join(" AND ")
        }
    }

    /// Combine two rules with logical AND.
    ///
    /// If both are pure equi-join rules, the result is an equi-join on the union
    /// of their columns (the fast path — no cross-join). Otherwise it falls back
    /// to a custom cross-join + filter.
    pub fn and(self, other: BlockingRule) -> Self {
        if self.predicate_dsl.is_none() && other.predicate_dsl.is_none() {
            let mut columns = self.columns;
            for c in other.columns {
                if !columns.contains(&c) {
                    columns.push(c);
                }
            }
            Self {
                columns,
                description: None,
                predicate_dsl: None,
            }
        } else {
            Self::custom(&format!(
                "({}) AND ({})",
                self.condition_sql(),
                other.condition_sql()
            ))
        }
    }

    /// Combine two rules with logical OR (cross-join + filter).
    pub fn or(self, other: BlockingRule) -> Self {
        Self::custom(&format!(
            "({}) OR ({})",
            self.condition_sql(),
            other.condition_sql()
        ))
    }

    /// Negate this rule (cross-join + filter).
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self::custom(&format!("NOT ({})", self.condition_sql()))
    }
}

/// Suffix all columns in a LazyFrame, adding `_l` or `_r`.
fn suffix_columns(lf: &LazyFrame, suffix: &str) -> LazyFrame {
    lf.clone().select([col("*").name().suffix(suffix)])
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
    lf: &LazyFrame,
    blocking_rules: &[BlockingRule],
    link_type: &LinkType,
    unique_id_col: &str,
    source_dataset_column: Option<&str>,
) -> Result<LazyFrame> {
    generate_blocked_pairs_impl(
        lf,
        lf,
        blocking_rules,
        link_type,
        unique_id_col,
        source_dataset_column,
        true,
    )
}

/// Generate candidate pairs by blocking records from `right_lf` against records
/// in `left_lf` (two distinct frames).
///
/// Unlike [`generate_blocked_pairs`], this produces **all** pairs satisfying a
/// blocking rule with no self-pair / id-ordering filtering — the two frames are
/// assumed to be distinct populations (e.g. existing records vs. new records in
/// [`Linker::find_matches_to_new_records`](crate::linker::Linker::find_matches_to_new_records)).
///
/// # Errors
///
/// Returns an error if a Polars join or schema operation fails, or if no
/// blocking rules are provided.
pub fn generate_blocked_pairs_between(
    left_lf: &LazyFrame,
    right_lf: &LazyFrame,
    blocking_rules: &[BlockingRule],
    link_type: &LinkType,
    unique_id_col: &str,
    source_dataset_column: Option<&str>,
) -> Result<LazyFrame> {
    generate_blocked_pairs_impl(
        left_lf,
        right_lf,
        blocking_rules,
        link_type,
        unique_id_col,
        source_dataset_column,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_blocked_pairs_impl(
    left_lf: &LazyFrame,
    right_lf: &LazyFrame,
    blocking_rules: &[BlockingRule],
    link_type: &LinkType,
    unique_id_col: &str,
    source_dataset_column: Option<&str>,
    same_frame: bool,
) -> Result<LazyFrame> {
    let uid_l = format!("{unique_id_col}_l");
    let uid_r = format!("{unique_id_col}_r");

    let mut left = suffix_columns(left_lf, "_l");
    let mut right = suffix_columns(right_lf, "_r");

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
        let joined = if let Some(dsl) = &rule.predicate_dsl {
            // Custom predicate: evaluate via a full cross-join + filter. This is
            // O(n²); the equi-join path below is preferred whenever possible.
            log::warn!(
                "Blocking rule {i} uses a custom predicate; this performs a full \
                 cross-join (O(n²)) and may be slow on large inputs."
            );
            let cond = polars::sql::sql_expr(dsl).map_err(|e| {
                crate::error::WeldrsError::Config(format!(
                    "Invalid blocking predicate SQL '{dsl}': {e}"
                ))
            })?;
            left.clone().cross_join(right.clone(), None).filter(cond)
        } else {
            // Equi-join on each blocking column.
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

            // Inner join drops the right key columns; re-add them from the left
            // keys (values are guaranteed equal by the join condition).
            for c in &rule.columns {
                joined = joined
                    .with_column(col(format!("{c}_l").as_str()).alias(format!("{c}_r").as_str()));
            }
            joined
        };

        // Filter out self-pairs and, for deduplication, keep only uid_l < uid_r.
        // Cross-frame blocking (`same_frame == false`) keeps every matched pair:
        // the two frames are distinct populations, so there are no self-pairs to
        // remove and no ordering to enforce.
        let filtered = if !same_frame {
            joined
        } else {
            match link_type {
                LinkType::DedupeOnly => joined.filter(col(uid_l.as_str()).lt(col(uid_r.as_str()))),
                LinkType::LinkOnly => {
                    if let Some(src_col) = source_dataset_column {
                        let src_l = format!("{src_col}_l");
                        let src_r = format!("{src_col}_r");
                        joined.filter(
                            col(src_l.as_str())
                                .neq(col(src_r.as_str()))
                                .and(col(uid_l.as_str()).lt(col(uid_r.as_str()))),
                        )
                    } else {
                        // Fallback: uid inequality when no source column is set.
                        joined.filter(col(uid_l.as_str()).neq(col(uid_r.as_str())))
                    }
                }
                LinkType::LinkAndDedupe => {
                    joined.filter(col(uid_l.as_str()).lt(col(uid_r.as_str())))
                }
            }
        };

        let with_key = filtered
            .with_column(lit(i as u32).alias("match_key"))
            .select(output_cols.clone());
        all_pairs.push(with_key);
    }

    if all_pairs.is_empty() {
        return Err(crate::error::WeldrsError::Config(
            "No blocking rules provided. At least one BlockingRule is required to generate \
             candidate pairs. If you truly need all-pairs comparison, add a blocking rule \
             on a column with a single constant value."
                .to_string(),
        ));
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

    fn small_lazy_frame() -> LazyFrame {
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
        let lf = small_lazy_frame();
        let rules = vec![BlockingRule::on(&["city"])];
        let pairs = generate_blocked_pairs(&lf, &rules, &LinkType::DedupeOnly, "unique_id", None)
            .unwrap()
            .collect()
            .unwrap();

        let uid_l = pairs.column("unique_id_l").unwrap().i64().unwrap();
        let uid_r = pairs.column("unique_id_r").unwrap().i64().unwrap();

        for (l, r) in uid_l.into_iter().zip(uid_r) {
            assert!(l.unwrap() < r.unwrap(), "Expected uid_l < uid_r");
        }
    }

    #[test]
    fn test_no_blocking_rules_errors() {
        let lf = small_lazy_frame();
        let result = generate_blocked_pairs(&lf, &[], &LinkType::DedupeOnly, "unique_id", None);
        assert!(
            result.is_err(),
            "Expected Config error when no blocking rules provided"
        );
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => unreachable!(),
        };
        assert!(
            err_msg.contains("blocking rule"),
            "Error should mention blocking rules, got: {err_msg}"
        );
    }

    #[test]
    fn test_multi_rule_deduplication() {
        let lf = small_lazy_frame();
        // Two rules that can produce overlapping pairs (city=London gives (1,2),(1,4),(2,4))
        let rules = vec![
            BlockingRule::on(&["city"]),
            BlockingRule::on(&["first_name"]),
        ];
        let pairs = generate_blocked_pairs(&lf, &rules, &LinkType::DedupeOnly, "unique_id", None)
            .unwrap()
            .collect()
            .unwrap();

        // Check no duplicate (uid_l, uid_r) pairs
        let n_before = pairs.height();
        let deduped = pairs
            .lazy()
            .unique(
                Some(cols(["unique_id_l", "unique_id_r"])),
                UniqueKeepStrategy::First,
            )
            .collect()
            .unwrap();
        assert_eq!(n_before, deduped.height());
    }

    #[test]
    fn test_match_key_assignment() {
        let lf = small_lazy_frame();
        let rules = vec![BlockingRule::on(&["city"])];
        let pairs = generate_blocked_pairs(&lf, &rules, &LinkType::DedupeOnly, "unique_id", None)
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
        let lf = small_lazy_frame();
        let rules = vec![BlockingRule::on(&["city"])];
        let pairs = generate_blocked_pairs(&lf, &rules, &LinkType::DedupeOnly, "unique_id", None)
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

    #[test]
    fn test_and_of_equi_rules_uses_union_columns() {
        // AND of two pure equi-join rules stays an equi-join (fast path).
        let rule = BlockingRule::on(&["city"]).and(BlockingRule::on(&["first_name"]));
        assert_eq!(rule.columns, vec!["city", "first_name"]);
        assert!(rule.predicate_dsl.is_none());
    }

    #[test]
    fn test_custom_rule_matches_equivalent_equi_join() {
        let lf = small_lazy_frame();
        let from_equi = generate_blocked_pairs(
            &lf,
            &[BlockingRule::on(&["city"])],
            &LinkType::DedupeOnly,
            "unique_id",
            None,
        )
        .unwrap()
        .collect()
        .unwrap();
        let from_custom = generate_blocked_pairs(
            &lf,
            &[BlockingRule::custom("city_l = city_r")],
            &LinkType::DedupeOnly,
            "unique_id",
            None,
        )
        .unwrap()
        .collect()
        .unwrap();

        // Same number of candidate pairs via either path.
        assert_eq!(from_equi.height(), from_custom.height());
    }

    #[test]
    fn test_or_rule_unions_conditions() {
        let lf = small_lazy_frame();
        // city OR first_name. Cross-join + filter.
        let rule = BlockingRule::on(&["city"]).or(BlockingRule::on(&["first_name"]));
        assert!(rule.predicate_dsl.is_some());
        let pairs = generate_blocked_pairs(&lf, &[rule], &LinkType::DedupeOnly, "unique_id", None)
            .unwrap()
            .collect()
            .unwrap();
        // Alice(1,3) share first_name; London(1,2),(1,4),(2,4) share city → at
        // least the city pairs plus the Alice pair are present.
        assert!(pairs.height() >= 4);
    }

    #[test]
    fn test_link_only_with_source_column() {
        let lf = df!(
            "unique_id" => [1i64, 2, 3, 101, 102, 103],
            "first_name" => ["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"],
            "city" => ["London", "London", "Paris", "London", "London", "Paris"],
            "source_dataset" => ["a", "a", "a", "b", "b", "b"],
        )
        .unwrap()
        .lazy();

        let rules = vec![BlockingRule::on(&["city"])];
        let pairs = generate_blocked_pairs(
            &lf,
            &rules,
            &LinkType::LinkOnly,
            "unique_id",
            Some("source_dataset"),
        )
        .unwrap()
        .collect()
        .unwrap();

        // Verify: all pairs should be cross-dataset (source_l != source_r)
        let src_l = pairs.column("source_dataset_l").unwrap().str().unwrap();
        let src_r = pairs.column("source_dataset_r").unwrap().str().unwrap();
        for (l, r) in src_l.into_iter().zip(src_r) {
            assert_ne!(l, r, "LinkOnly pairs must be cross-dataset");
        }
        assert!(pairs.height() > 0, "Should produce cross-dataset pairs");
    }
}
