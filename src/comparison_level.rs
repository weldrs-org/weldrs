//! Comparison predicates and levels.
//!
//! A [`ComparisonLevel`] pairs a [`ComparisonPredicate`] (the rule that
//! decides whether two values "agree") with trained m/u probabilities.
//! Levels are stacked inside a [`Comparison`](crate::comparison::Comparison)
//! and evaluated top-to-bottom; the first matching level wins.
//!
//! # Available predicates
//!
//! | Variant | Meaning |
//! |---------|---------|
//! | [`ComparisonPredicate::NullCheck`] | Both values are null |
//! | [`ComparisonPredicate::ExactMatch`] | Values are exactly equal |
//! | [`ComparisonPredicate::LevenshteinDistance`] | Edit distance ≤ threshold |
//! | [`ComparisonPredicate::JaroWinklerSimilarity`] | Jaro-Winkler ≥ threshold |
//! | [`ComparisonPredicate::JaroSimilarity`] | Jaro ≥ threshold |
//! | [`ComparisonPredicate::Else`] | Catch-all for remaining pairs |
//!
//! Most users will not construct [`ComparisonLevel`] values directly — use
//! [`ComparisonBuilder`](crate::comparison::ComparisonBuilder) instead, which
//! handles level ordering and default m/u assignment automatically.

use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, WeldrsError};

/// Apply a string predicate using unique-value deduplication.
///
/// Instead of computing the predicate for every row, this function:
/// 1. Identifies the set of unique (left, right) value pairs
/// 2. Computes the predicate in parallel over only the unique pairs
/// 3. Maps results back to all rows via lookup
///
/// This is highly effective when many rows share the same value pairs
/// (e.g., after blocking on surname with ~50 unique names per side,
/// 500K pairs may yield only ~2,500 unique combinations).
///
/// Includes a heuristic fallback: if unique pairs exceed 50% of total
/// non-null rows, falls back to direct per-row computation to avoid
/// HashMap overhead when values are highly unique.
fn par_pairwise_string_predicate(
    s: &Column,
    col_l_key: &PlSmallStr,
    col_r_key: &PlSmallStr,
    predicate: impl Fn(&str, &str) -> bool + Send + Sync,
) -> PolarsResult<Column> {
    let ca = s.struct_()?;
    let left_str = ca.field_by_name(col_l_key)?.str()?.clone();
    let right_str = ca.field_by_name(col_r_key)?.str()?.clone();
    let n = left_str.len();

    // Phase 1: Identify unique value pairs and map each row to its pair index.
    let mut pair_to_idx: HashMap<(&str, &str), u32> = HashMap::new();
    let mut unique_pairs: Vec<(&str, &str)> = Vec::new();
    let mut row_pair_idx: Vec<u32> = Vec::with_capacity(n);
    let mut non_null_count: usize = 0;

    for i in 0..n {
        match (left_str.get(i), right_str.get(i)) {
            (Some(l), Some(r)) => {
                non_null_count += 1;
                let next_idx = unique_pairs.len() as u32;
                let idx = *pair_to_idx.entry((l, r)).or_insert_with(|| {
                    unique_pairs.push((l, r));
                    next_idx
                });
                row_pair_idx.push(idx);
            }
            _ => {
                // Sentinel: u32::MAX marks null rows
                row_pair_idx.push(u32::MAX);
            }
        }
    }

    // Heuristic: if unique pairs > 50% of non-null rows, fall back to direct
    // per-row computation to avoid HashMap overhead for highly unique data.
    if non_null_count > 0 && unique_pairs.len() * 2 > non_null_count {
        let bools: Vec<bool> = (0..n)
            .into_par_iter()
            .map(|i| match (left_str.get(i), right_str.get(i)) {
                (Some(l), Some(r)) => predicate(l, r),
                _ => false,
            })
            .collect();
        let out = BooleanChunked::from_iter(bools.into_iter().map(Some));
        return Ok(out.into_column());
    }

    // Phase 2: Compute predicate for each unique pair in parallel.
    let pair_results: Vec<bool> = unique_pairs
        .par_iter()
        .map(|(l, r)| predicate(l, r))
        .collect();

    // Phase 3: Map results back to all rows.
    let bools: Vec<bool> = row_pair_idx
        .iter()
        .map(|&idx| {
            if idx == u32::MAX {
                false
            } else {
                pair_results[idx as usize]
            }
        })
        .collect();

    let out = BooleanChunked::from_iter(bools.into_iter().map(Some));
    Ok(out.into_column())
}

/// A predicate that defines how two records are compared at a single level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonPredicate {
    /// Both left and right values are null.
    NullCheck {
        /// Column name to check for nulls.
        col: String,
    },
    /// Left and right values are exactly equal.
    ExactMatch {
        /// Column name to compare.
        col: String,
    },
    /// Levenshtein edit distance is at most `threshold`.
    LevenshteinDistance {
        /// Column name to compare.
        col: String,
        /// Maximum edit distance.
        threshold: u32,
    },
    /// Jaro-Winkler similarity is at least `threshold`.
    JaroWinklerSimilarity {
        /// Column name to compare.
        col: String,
        /// Minimum similarity score (0.0–1.0).
        threshold: f64,
    },
    /// Jaro similarity is at least `threshold`.
    JaroSimilarity {
        /// Column name to compare.
        col: String,
        /// Minimum similarity score (0.0–1.0).
        threshold: f64,
    },
    /// Catch-all level for all remaining pairs.
    Else,
}

impl ComparisonPredicate {
    /// The input column name this predicate operates on (if any).
    pub fn column(&self) -> Option<&str> {
        match self {
            Self::NullCheck { col }
            | Self::ExactMatch { col }
            | Self::LevenshteinDistance { col, .. }
            | Self::JaroWinklerSimilarity { col, .. }
            | Self::JaroSimilarity { col, .. } => Some(col),
            Self::Else => None,
        }
    }

    /// Build a Polars boolean expression for this predicate.
    ///
    /// Column names in the blocked-pairs DataFrame are expected to be
    /// `{col}_l` and `{col}_r`.
    pub fn to_expr(&self) -> Result<Expr> {
        match self {
            Self::NullCheck { col: c } => {
                let l = col(format!("{c}_l"));
                let r = col(format!("{c}_r"));
                Ok(l.is_null().and(r.is_null()))
            }
            Self::ExactMatch { col: c } => {
                let l = col(format!("{c}_l"));
                let r = col(format!("{c}_r"));
                Ok(l.eq(r))
            }
            Self::LevenshteinDistance { col: c, threshold } => {
                let threshold = *threshold;
                let col_l = format!("{c}_l");
                let col_r = format!("{c}_r");
                let col_l_key = PlSmallStr::from(col_l.as_str());
                let col_r_key = PlSmallStr::from(col_r.as_str());
                Ok(as_struct(vec![col(&col_l), col(&col_r)])
                    .map(
                        move |s: Column| {
                            par_pairwise_string_predicate(&s, &col_l_key, &col_r_key, |l, r| {
                                crate::string_distance::levenshtein_within(l, r, threshold)
                            })
                            .map(Some)
                        },
                        GetOutput::from_type(DataType::Boolean),
                    )
                    .with_fmt("levenshtein_distance"))
            }
            Self::JaroWinklerSimilarity { col: c, threshold } => {
                let threshold = *threshold;
                let col_l = format!("{c}_l");
                let col_r = format!("{c}_r");
                let col_l_key = PlSmallStr::from(col_l.as_str());
                let col_r_key = PlSmallStr::from(col_r.as_str());
                Ok(as_struct(vec![col(&col_l), col(&col_r)])
                    .map(
                        move |s: Column| {
                            par_pairwise_string_predicate(&s, &col_l_key, &col_r_key, |l, r| {
                                crate::string_distance::jaro_winkler_similarity(l, r) >= threshold
                            })
                            .map(Some)
                        },
                        GetOutput::from_type(DataType::Boolean),
                    )
                    .with_fmt("jaro_winkler_similarity"))
            }
            Self::JaroSimilarity { col: c, threshold } => {
                let threshold = *threshold;
                let col_l = format!("{c}_l");
                let col_r = format!("{c}_r");
                let col_l_key = PlSmallStr::from(col_l.as_str());
                let col_r_key = PlSmallStr::from(col_r.as_str());
                Ok(as_struct(vec![col(&col_l), col(&col_r)])
                    .map(
                        move |s: Column| {
                            par_pairwise_string_predicate(&s, &col_l_key, &col_r_key, |l, r| {
                                crate::string_distance::jaro_similarity(l, r) >= threshold
                            })
                            .map(Some)
                        },
                        GetOutput::from_type(DataType::Boolean),
                    )
                    .with_fmt("jaro_similarity"))
            }
            Self::Else => Err(WeldrsError::Config(
                "Else predicate has no expression; it is the catch-all".into(),
            )),
        }
    }
}

/// A single level within a [`Comparison`](crate::comparison::Comparison).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonLevel {
    /// The rule that decides whether two values agree at this level.
    pub predicate: ComparisonPredicate,
    /// Human-readable label for this level (e.g. "Exact match").
    pub label: String,
    /// Whether this level represents the "both values are null" case.
    /// Null levels are neutral — they contribute a Bayes factor of 1.0.
    pub is_null_level: bool,
    /// Integer identifier assigned by the parent `Comparison` during
    /// construction. Higher values indicate stronger agreement.
    pub comparison_vector_value: i32,
    /// Probability of this level agreeing given the records **are** a match.
    pub m_probability: Option<f64>,
    /// Probability of this level agreeing given the records **are not** a match.
    pub u_probability: Option<f64>,
    /// If true, EM will not update this level's m-probability.
    pub fix_m_probability: bool,
    /// If true, EM will not update this level's u-probability.
    pub fix_u_probability: bool,
}

impl ComparisonLevel {
    /// Bayes factor for this level: m / u.
    pub fn bayes_factor(&self) -> Option<f64> {
        match (self.m_probability, self.u_probability) {
            (Some(m), Some(u)) => {
                if u == 0.0 {
                    Some(f64::INFINITY)
                } else {
                    Some(m / u)
                }
            }
            _ => None,
        }
    }

    /// Log2 of the Bayes factor (the "match weight" for this level).
    pub fn log2_bayes_factor(&self) -> Option<f64> {
        self.bayes_factor().map(f64::log2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a paired DF, apply the predicate expression, collect the boolean result.
    fn eval_predicate(
        predicate: &ComparisonPredicate,
        vals_l: &[&str],
        vals_r: &[&str],
    ) -> Vec<Option<bool>> {
        let col_name = predicate.column().unwrap();
        let col_l = format!("{col_name}_l");
        let col_r = format!("{col_name}_r");
        let df = df!(
            &col_l => vals_l,
            &col_r => vals_r,
        )
        .unwrap();
        let expr = predicate.to_expr().unwrap();
        let result = df
            .lazy()
            .with_column(expr.alias("result"))
            .collect()
            .unwrap();
        result
            .column("result")
            .unwrap()
            .bool()
            .unwrap()
            .into_iter()
            .collect()
    }

    /// Helper: build a paired DF with nullable Utf8 columns.
    fn eval_predicate_nullable(
        predicate: &ComparisonPredicate,
        vals_l: &[Option<&str>],
        vals_r: &[Option<&str>],
    ) -> Vec<Option<bool>> {
        let col_name = predicate.column().unwrap();
        let col_l = format!("{col_name}_l");
        let col_r = format!("{col_name}_r");
        let df = df!(
            &col_l => vals_l,
            &col_r => vals_r,
        )
        .unwrap();
        let expr = predicate.to_expr().unwrap();
        let result = df
            .lazy()
            .with_column(expr.alias("result"))
            .collect()
            .unwrap();
        result
            .column("result")
            .unwrap()
            .bool()
            .unwrap()
            .into_iter()
            .collect()
    }

    #[test]
    fn test_null_check_to_expr() {
        let pred = ComparisonPredicate::NullCheck { col: "name".into() };
        let results = eval_predicate_nullable(
            &pred,
            &[None, Some("Alice"), None],
            &[None, Some("Bob"), Some("Carol")],
        );
        // Both null → true, one non-null → false, mixed → false
        assert_eq!(results, vec![Some(true), Some(false), Some(false)]);
    }

    #[test]
    fn test_exact_match_to_expr() {
        let pred = ComparisonPredicate::ExactMatch { col: "name".into() };
        // Equal values → true, different → false
        let results = eval_predicate(&pred, &["Alice", "Alice"], &["Alice", "Bob"]);
        assert_eq!(results, vec![Some(true), Some(false)]);

        // null == null → null in Polars equality
        let results_null = eval_predicate_nullable(&pred, &[None], &[None]);
        assert_eq!(results_null, vec![None]);
    }

    #[test]
    fn test_levenshtein_to_expr() {
        let pred = ComparisonPredicate::LevenshteinDistance {
            col: "name".into(),
            threshold: 1,
        };
        // "kitten"→"sitten" distance=1 → true; "kitten"→"sitting" distance=3 → false
        let results = eval_predicate(&pred, &["kitten", "kitten"], &["sitten", "sitting"]);
        assert_eq!(results, vec![Some(true), Some(false)]);
    }

    #[test]
    fn test_jaro_winkler_to_expr() {
        let pred = ComparisonPredicate::JaroWinklerSimilarity {
            col: "name".into(),
            threshold: 0.85,
        };
        // "martha"→"marhta" JW ≈ 0.96 → true; "abc"→"xyz" → false
        let results = eval_predicate(&pred, &["martha", "abc"], &["marhta", "xyz"]);
        assert_eq!(results[0], Some(true));
        assert_eq!(results[1], Some(false));
    }

    #[test]
    fn test_jaro_to_expr() {
        let pred = ComparisonPredicate::JaroSimilarity {
            col: "name".into(),
            threshold: 0.8,
        };
        // "martha"→"marhta" Jaro ≈ 0.94 → true
        let results = eval_predicate(&pred, &["martha"], &["marhta"]);
        assert_eq!(results[0], Some(true));
    }

    #[test]
    fn test_else_to_expr_errors() {
        let pred = ComparisonPredicate::Else;
        let result = pred.to_expr();
        assert!(result.is_err());
        match result.unwrap_err() {
            WeldrsError::Config(_) => {} // expected
            other => panic!("Expected Config error, got: {other:?}"),
        }
    }

    #[test]
    fn test_bayes_factor_normal() {
        let level = ComparisonLevel {
            predicate: ComparisonPredicate::Else,
            label: "test".into(),
            is_null_level: false,
            comparison_vector_value: 0,
            m_probability: Some(0.9),
            u_probability: Some(0.1),
            fix_m_probability: false,
            fix_u_probability: false,
        };
        let bf = level.bayes_factor().unwrap();
        assert!((bf - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_bayes_factor_u_zero() {
        let level = ComparisonLevel {
            predicate: ComparisonPredicate::Else,
            label: "test".into(),
            is_null_level: false,
            comparison_vector_value: 0,
            m_probability: Some(0.9),
            u_probability: Some(0.0),
            fix_m_probability: false,
            fix_u_probability: false,
        };
        assert_eq!(level.bayes_factor(), Some(f64::INFINITY));
    }

    #[test]
    fn test_bayes_factor_none_when_missing() {
        let level_no_m = ComparisonLevel {
            predicate: ComparisonPredicate::Else,
            label: "test".into(),
            is_null_level: false,
            comparison_vector_value: 0,
            m_probability: None,
            u_probability: Some(0.1),
            fix_m_probability: false,
            fix_u_probability: false,
        };
        assert_eq!(level_no_m.bayes_factor(), None);

        let level_no_u = ComparisonLevel {
            predicate: ComparisonPredicate::Else,
            label: "test".into(),
            is_null_level: false,
            comparison_vector_value: 0,
            m_probability: Some(0.9),
            u_probability: None,
            fix_m_probability: false,
            fix_u_probability: false,
        };
        assert_eq!(level_no_u.bayes_factor(), None);
    }

    #[test]
    fn test_levenshtein_parallel_correctness() {
        let n = 10_000;
        let left: Vec<String> = (0..n).map(|i| format!("name_{i}")).collect();
        let right: Vec<String> = (0..n).map(|i| format!("namx_{i}")).collect();
        let threshold = 2u32;

        let pred = ComparisonPredicate::LevenshteinDistance {
            col: "name".into(),
            threshold,
        };

        let left_refs: Vec<&str> = left.iter().map(|s| s.as_str()).collect();
        let right_refs: Vec<&str> = right.iter().map(|s| s.as_str()).collect();
        let results = eval_predicate(&pred, &left_refs, &right_refs);

        // Verify against direct strsim computation.
        for (i, result) in results.iter().enumerate() {
            let expected = strsim::levenshtein(&left[i], &right[i]) as u32 <= threshold;
            assert_eq!(*result, Some(expected), "Mismatch at index {i}");
        }
    }

    #[test]
    fn test_jaro_winkler_parallel_correctness() {
        let n = 10_000;
        let left: Vec<String> = (0..n).map(|i| format!("alice_{i}")).collect();
        let right: Vec<String> = (0..n).map(|i| format!("alicx_{i}")).collect();
        let threshold = 0.8;

        let pred = ComparisonPredicate::JaroWinklerSimilarity {
            col: "name".into(),
            threshold,
        };

        let left_refs: Vec<&str> = left.iter().map(|s| s.as_str()).collect();
        let right_refs: Vec<&str> = right.iter().map(|s| s.as_str()).collect();
        let results = eval_predicate(&pred, &left_refs, &right_refs);

        for (i, result) in results.iter().enumerate() {
            let expected = strsim::jaro_winkler(&left[i], &right[i]) >= threshold;
            assert_eq!(*result, Some(expected), "Mismatch at index {i}");
        }
    }

    #[test]
    fn test_string_similarity_large_input() {
        let n = 100_000;
        let left: Vec<String> = (0..n).map(|i| format!("record_{i}")).collect();
        let right: Vec<String> = (0..n).map(|i| format!("recxrd_{i}")).collect();

        let pred = ComparisonPredicate::JaroWinklerSimilarity {
            col: "name".into(),
            threshold: 0.85,
        };

        let left_refs: Vec<&str> = left.iter().map(|s| s.as_str()).collect();
        let right_refs: Vec<&str> = right.iter().map(|s| s.as_str()).collect();
        let results = eval_predicate(&pred, &left_refs, &right_refs);

        assert_eq!(results.len(), n);
        // All results should be Some (no panics or data races).
        assert!(results.iter().all(|r| r.is_some()));
    }
}
