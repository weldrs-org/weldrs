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
//! | [`ComparisonPredicate::DamerauLevenshtein`] | Edit distance with transpositions ≤ threshold |
//! | [`ComparisonPredicate::Hamming`] | Differing positions ≤ threshold (equal-length only) |
//! | [`ComparisonPredicate::JaroWinklerSimilarity`] | Jaro-Winkler ≥ threshold |
//! | [`ComparisonPredicate::JaroSimilarity`] | Jaro ≥ threshold |
//! | [`ComparisonPredicate::Jaccard`] | Character-set Jaccard ≥ threshold |
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

/// Build a Polars boolean expression that applies a per-pair string predicate
/// to the suffixed `{col}_l` / `{col}_r` columns, using the unique-pair
/// deduplication fast path in [`par_pairwise_string_predicate`].
///
/// Centralises the `as_struct(...).map_with_fmt_str(...)` boilerplate shared by
/// every string-distance predicate variant.
fn string_predicate_expr(
    col_name: &str,
    output_name: &'static str,
    predicate: impl Fn(&str, &str) -> bool + Send + Sync + 'static,
) -> Expr {
    let col_l = format!("{col_name}_l");
    let col_r = format!("{col_name}_r");
    let col_l_key = PlSmallStr::from(col_l.as_str());
    let col_r_key = PlSmallStr::from(col_r.as_str());
    as_struct(vec![col(&col_l), col(&col_r)]).map_with_fmt_str(
        move |s: Column| par_pairwise_string_predicate(&s, &col_l_key, &col_r_key, &predicate),
        move |_schema, _field| Ok(Field::new(output_name.into(), DataType::Boolean)),
        output_name,
    )
}

/// Build an expression that returns `true` when the cosine similarity of two
/// numeric list columns (`{col}_l`, `{col}_r`) is at least `threshold`.
///
/// Cosine similarity has no native Polars list reduction, so this evaluates a
/// per-row closure over the two list columns. Rows where either vector is null,
/// empty, length-mismatched, or zero-magnitude are treated as non-matches.
fn cosine_similarity_expr(col_name: &str, threshold: f64) -> Expr {
    let col_l = format!("{col_name}_l");
    let col_r = format!("{col_name}_r");
    let col_l_key = PlSmallStr::from(col_l.as_str());
    let col_r_key = PlSmallStr::from(col_r.as_str());
    as_struct(vec![col(&col_l), col(&col_r)]).map_with_fmt_str(
        move |s: Column| -> PolarsResult<Column> {
            let ca = s.struct_()?;
            let left = ca.field_by_name(&col_l_key)?;
            let right = ca.field_by_name(&col_r_key)?;
            let ll = left.list()?;
            let rl = right.list()?;
            let n = ll.len();
            let mut out: Vec<bool> = Vec::with_capacity(n);
            for i in 0..n {
                let pair = (ll.get_as_series(i), rl.get_as_series(i));
                let is_match = match pair {
                    (Some(a), Some(b)) => {
                        let a = a.cast(&DataType::Float64)?;
                        let b = b.cast(&DataType::Float64)?;
                        let av = a.f64()?;
                        let bv = b.f64()?;
                        cosine_at_least(av, bv, threshold)
                    }
                    _ => false,
                };
                out.push(is_match);
            }
            Ok(BooleanChunked::from_iter(out.into_iter().map(Some)).into_column())
        },
        move |_schema, _field| Ok(Field::new("cosine_similarity".into(), DataType::Boolean)),
        "cosine_similarity",
    )
}

/// Whether the cosine similarity of two equal-length f64 vectors is `>= thresh`.
/// Returns `false` for length mismatch, empties, nulls, or zero magnitude.
fn cosine_at_least(a: &Float64Chunked, b: &Float64Chunked, thresh: f64) -> bool {
    if a.len() != b.len() || a.is_empty() {
        return false;
    }
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (x, y) in a.into_iter().zip(b) {
        match (x, y) {
            (Some(x), Some(y)) => {
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            _ => return false,
        }
    }
    if na == 0.0 || nb == 0.0 {
        return false;
    }
    (dot / (na.sqrt() * nb.sqrt())) >= thresh
}

/// Apply a string predicate using unique-value deduplication.
///
/// Instead of computing the predicate for every row, this function:
/// 1. Identifies the set of unique (left, right) value pairs
/// 2. Computes the predicate in parallel over only the unique pairs
/// 3. Maps results back to all rows via lookup
///
/// This is highly effective when many rows share the same value pairs
/// (e.g., after blocking on last_name with ~50 unique names per side,
/// 500K pairs may yield only ~2,500 unique combinations).
///
/// Includes a heuristic fallback: if unique pairs exceed 50% of total
/// non-null rows, falls back to direct per-row computation to avoid
/// HashMap overhead when values are highly unique.
///
/// # Null handling
///
/// Rows where either value is null are tracked using `u32::MAX` as a
/// sentinel index in `row_pair_idx`. This limits the maximum number of
/// distinct value pairs to `u32::MAX - 1` (≈4 billion), which is far
/// beyond practical working set sizes after blocking.
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

    // Phase 3: Map results back to all rows (parallel — read-only lookups).
    let bools: Vec<bool> = row_pair_idx
        .par_iter()
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

/// Unit for [`ComparisonPredicate::AbsoluteDateDifference`].
///
/// `Day` compares the absolute difference in days-since-epoch; `Month` and
/// `Year` compare calendar-component differences (e.g. `2020-12` vs `2021-01`
/// is one month apart), matching the boundary-counting semantics of SQL
/// `date_diff`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DateMetric {
    /// Difference in days.
    Day,
    /// Difference in calendar months (`year*12 + month`).
    Month,
    /// Difference in calendar years.
    Year,
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
    /// Damerau-Levenshtein edit distance (with transpositions) is at most
    /// `threshold`.
    DamerauLevenshtein {
        /// Column name to compare.
        col: String,
        /// Maximum edit distance.
        threshold: u32,
    },
    /// Hamming distance is at most `threshold` (unequal-length strings never
    /// match).
    Hamming {
        /// Column name to compare.
        col: String,
        /// Maximum number of differing positions.
        threshold: u32,
    },
    /// Jaccard similarity of the character sets is at least `threshold`.
    Jaccard {
        /// Column name to compare.
        col: String,
        /// Minimum similarity score (0.0–1.0).
        threshold: f64,
    },
    /// Absolute date difference is at most `threshold` in the given
    /// [`DateMetric`]. The column must be a Date dtype.
    AbsoluteDateDifference {
        /// Date column to compare.
        col: String,
        /// Maximum allowed difference (inclusive).
        threshold: i64,
        /// Unit of the threshold.
        metric: DateMetric,
    },
    /// Relative numeric difference `|l - r| / max(|l|, |r|)` is at most
    /// `threshold` (two zeros count as a match).
    PercentageDifference {
        /// Numeric column to compare.
        col: String,
        /// Maximum allowed relative difference (e.g. `0.1` for 10%).
        threshold: f64,
    },
    /// Great-circle (Haversine) distance between two lat/long points is at most
    /// `threshold_km` kilometres.
    DistanceInKm {
        /// Latitude column (degrees).
        lat_col: String,
        /// Longitude column (degrees).
        long_col: String,
        /// Maximum allowed distance in kilometres.
        threshold_km: f64,
    },
    /// The two list columns share at least `min_size` elements.
    ArrayIntersect {
        /// List column to compare.
        col: String,
        /// Minimum required intersection size.
        min_size: u32,
    },
    /// Cosine similarity between two equal-length numeric vectors (list columns)
    /// is at least `threshold`.
    CosineSimilarity {
        /// List-of-float column to compare.
        col: String,
        /// Minimum cosine similarity (0.0–1.0).
        threshold: f64,
    },
    /// Logical AND of two or more sub-predicates (all must hold).
    And(
        /// Sub-predicates that must all hold.
        Vec<Box<ComparisonPredicate>>,
    ),
    /// Logical OR of two or more sub-predicates (any may hold).
    Or(
        /// Sub-predicates, any of which may hold.
        Vec<Box<ComparisonPredicate>>,
    ),
    /// Logical negation of a sub-predicate.
    Not(
        /// The negated sub-predicate.
        Box<ComparisonPredicate>,
    ),
    /// A user-supplied boolean condition written in Polars SQL, referencing the
    /// suffixed `{col}_l` / `{col}_r` columns of the blocked-pairs frame
    /// (e.g. `"first_name_l = first_name_r AND city_l = city_r"`).
    ///
    /// This is the Rust analog of Splink's `CustomLevel(sql_condition)`. The
    /// expression is parsed lazily in [`to_expr`](Self::to_expr); a parse
    /// failure surfaces as [`WeldrsError::Config`].
    CustomPredicate {
        /// Boolean SQL expression over the suffixed columns.
        dsl: String,
    },
    /// Catch-all level for all remaining pairs.
    Else,
}

impl ComparisonPredicate {
    /// The input column name(s) this predicate operates on.
    ///
    /// Returns all referenced columns (composition predicates recurse into
    /// their children). The catch-all [`Else`](Self::Else) returns an empty
    /// list.
    pub fn columns(&self) -> Vec<&str> {
        match self {
            Self::NullCheck { col }
            | Self::ExactMatch { col }
            | Self::LevenshteinDistance { col, .. }
            | Self::DamerauLevenshtein { col, .. }
            | Self::Hamming { col, .. }
            | Self::Jaccard { col, .. }
            | Self::JaroWinklerSimilarity { col, .. }
            | Self::JaroSimilarity { col, .. }
            | Self::AbsoluteDateDifference { col, .. }
            | Self::PercentageDifference { col, .. }
            | Self::ArrayIntersect { col, .. }
            | Self::CosineSimilarity { col, .. } => vec![col.as_str()],
            Self::DistanceInKm {
                lat_col, long_col, ..
            } => vec![lat_col.as_str(), long_col.as_str()],
            Self::And(preds) | Self::Or(preds) => preds.iter().flat_map(|p| p.columns()).collect(),
            Self::Not(pred) => pred.columns(),
            // Columns inside a raw SQL condition are not statically tracked.
            Self::CustomPredicate { .. } => vec![],
            Self::Else => vec![],
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
                Ok(string_predicate_expr(
                    c,
                    "levenshtein_distance",
                    move |l, r| crate::string_distance::levenshtein_within(l, r, threshold),
                ))
            }
            Self::DamerauLevenshtein { col: c, threshold } => {
                let threshold = *threshold;
                Ok(string_predicate_expr(
                    c,
                    "damerau_levenshtein_distance",
                    move |l, r| crate::string_distance::damerau_levenshtein_within(l, r, threshold),
                ))
            }
            Self::Hamming { col: c, threshold } => {
                let threshold = *threshold;
                Ok(string_predicate_expr(c, "hamming_distance", move |l, r| {
                    crate::string_distance::hamming_within(l, r, threshold)
                }))
            }
            Self::Jaccard { col: c, threshold } => {
                let threshold = *threshold;
                Ok(string_predicate_expr(
                    c,
                    "jaccard_similarity",
                    move |l, r| crate::string_distance::jaccard_similarity(l, r) >= threshold,
                ))
            }
            Self::JaroWinklerSimilarity { col: c, threshold } => {
                let threshold = *threshold;
                Ok(string_predicate_expr(
                    c,
                    "jaro_winkler_similarity",
                    move |l, r| crate::string_distance::jaro_winkler_similarity(l, r) >= threshold,
                ))
            }
            Self::JaroSimilarity { col: c, threshold } => {
                let threshold = *threshold;
                Ok(string_predicate_expr(c, "jaro_similarity", move |l, r| {
                    crate::string_distance::jaro_similarity(l, r) >= threshold
                }))
            }
            Self::AbsoluteDateDifference {
                col: c,
                threshold,
                metric,
            } => {
                let threshold = *threshold;
                let l = col(format!("{c}_l"));
                let r = col(format!("{c}_r"));
                let diff = match metric {
                    DateMetric::Day => {
                        // Date casts to days-since-epoch (Int32).
                        (l.cast(DataType::Int64) - r.cast(DataType::Int64)).abs()
                    }
                    DateMetric::Month => {
                        let lm =
                            l.clone().dt().year() * lit(12) + l.dt().month().cast(DataType::Int32);
                        let rm =
                            r.clone().dt().year() * lit(12) + r.dt().month().cast(DataType::Int32);
                        (lm.cast(DataType::Int64) - rm.cast(DataType::Int64)).abs()
                    }
                    DateMetric::Year => (l.dt().year().cast(DataType::Int64)
                        - r.dt().year().cast(DataType::Int64))
                    .abs(),
                };
                Ok(diff.lt_eq(lit(threshold)))
            }
            Self::PercentageDifference { col: c, threshold } => {
                let threshold = *threshold;
                let l = col(format!("{c}_l"));
                let r = col(format!("{c}_r"));
                let num = (l.clone() - r.clone()).abs();
                // Denominator = max(|l|, |r|).
                let la = l.abs();
                let ra = r.abs();
                let denom = when(la.clone().gt_eq(ra.clone())).then(la).otherwise(ra);
                // Both zero → identical → match; otherwise compare the ratio.
                Ok(when(denom.clone().eq(lit(0.0)))
                    .then(lit(true))
                    .otherwise((num / denom).lt_eq(lit(threshold))))
            }
            Self::DistanceInKm {
                lat_col,
                long_col,
                threshold_km,
            } => {
                let threshold_km = *threshold_km;
                const DEG2RAD: f64 = std::f64::consts::PI / 180.0;
                let lat_l = col(format!("{lat_col}_l")) * lit(DEG2RAD);
                let lat_r = col(format!("{lat_col}_r")) * lit(DEG2RAD);
                let lon_l = col(format!("{long_col}_l")) * lit(DEG2RAD);
                let lon_r = col(format!("{long_col}_r")) * lit(DEG2RAD);

                let dlat_half = (lat_r.clone() - lat_l.clone()) / lit(2.0);
                let dlon_half = (lon_r - lon_l) / lit(2.0);
                let sin_dlat = dlat_half.sin();
                let sin_dlon = dlon_half.sin();
                let a = sin_dlat.clone() * sin_dlat
                    + lat_l.cos() * lat_r.cos() * sin_dlon.clone() * sin_dlon;
                // c = 2 * asin(min(1, sqrt(a)))  (clamp guards FP overshoot)
                let c = lit(2.0) * a.sqrt().clip_max(lit(1.0)).arcsin();
                let dist_km = lit(6371.0_f64) * c;
                Ok(dist_km.lt_eq(lit(threshold_km)))
            }
            Self::ArrayIntersect { col: c, min_size } => {
                let min_size = *min_size;
                let l = col(format!("{c}_l"));
                let r = col(format!("{c}_r"));
                Ok(l.list()
                    .set_intersection(r)
                    .list()
                    .len()
                    .gt_eq(lit(min_size)))
            }
            Self::CosineSimilarity { col: c, threshold } => {
                Ok(cosine_similarity_expr(c, *threshold))
            }
            Self::And(preds) => {
                let mut acc: Option<Expr> = None;
                for p in preds {
                    let e = p.to_expr()?;
                    acc = Some(match acc {
                        Some(prev) => prev.and(e),
                        None => e,
                    });
                }
                // An empty AND is vacuously true.
                Ok(acc.unwrap_or_else(|| lit(true)))
            }
            Self::Or(preds) => {
                let mut acc: Option<Expr> = None;
                for p in preds {
                    let e = p.to_expr()?;
                    acc = Some(match acc {
                        Some(prev) => prev.or(e),
                        None => e,
                    });
                }
                // An empty OR is vacuously false.
                Ok(acc.unwrap_or_else(|| lit(false)))
            }
            Self::Not(pred) => Ok(pred.to_expr()?.not()),
            Self::CustomPredicate { dsl } => polars::sql::sql_expr(dsl).map_err(|e| {
                WeldrsError::Config(format!("Invalid custom predicate SQL '{dsl}': {e}"))
            }),
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
        let col_name = predicate.columns()[0];
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
        let col_name = predicate.columns()[0];
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
    fn test_damerau_levenshtein_to_expr() {
        let pred = ComparisonPredicate::DamerauLevenshtein {
            col: "name".into(),
            threshold: 1,
        };
        // "teh"→"the" is a single transposition (DL=1) → true;
        // "abc"→"xyz" → false
        let results = eval_predicate(&pred, &["teh", "abc"], &["the", "xyz"]);
        assert_eq!(results, vec![Some(true), Some(false)]);
    }

    #[test]
    fn test_hamming_to_expr() {
        let pred = ComparisonPredicate::Hamming {
            col: "name".into(),
            threshold: 2,
        };
        // "karolin"→"kathrin" differs in 3 positions → false at threshold 2;
        // "abcd"→"abce" differs in 1 → true;
        // "abc"→"abcd" different lengths → false (Hamming undefined)
        let results = eval_predicate(
            &pred,
            &["karolin", "abcd", "abc"],
            &["kathrin", "abce", "abcd"],
        );
        assert_eq!(results, vec![Some(false), Some(true), Some(false)]);
    }

    #[test]
    fn test_jaccard_to_expr() {
        let pred = ComparisonPredicate::Jaccard {
            col: "name".into(),
            threshold: 0.5,
        };
        // identical → 1.0 ≥ 0.5 → true;
        // {a,b,c} vs {b,c,d} → 2/4 = 0.5 ≥ 0.5 → true;
        // {a,b,c} vs {x,y,z} → 0/6 = 0.0 → false
        let results = eval_predicate(&pred, &["abc", "abc", "abc"], &["abc", "bcd", "xyz"]);
        assert_eq!(results, vec![Some(true), Some(true), Some(false)]);
    }

    #[test]
    fn test_columns_returns_referenced_column() {
        let pred = ComparisonPredicate::Jaccard {
            col: "surname".into(),
            threshold: 0.7,
        };
        assert_eq!(pred.columns(), vec!["surname"]);
        // Else references no columns.
        assert!(ComparisonPredicate::Else.columns().is_empty());
    }

    /// Evaluate a predicate against a manually-built DataFrame, returning the
    /// boolean result column.
    fn eval_on_df(pred: &ComparisonPredicate, df: DataFrame) -> Vec<Option<bool>> {
        let expr = pred.to_expr().unwrap();
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
    fn test_percentage_difference_to_expr() {
        let pred = ComparisonPredicate::PercentageDifference {
            col: "amount".into(),
            threshold: 0.1,
        };
        // 100 vs 105 → 5/105 ≈ 0.048 ≤ 0.1 → true
        // 100 vs 200 → 100/200 = 0.5 > 0.1 → false
        // 0 vs 0 → identical → true
        let df = df!(
            "amount_l" => [100.0, 100.0, 0.0],
            "amount_r" => [105.0, 200.0, 0.0],
        )
        .unwrap();
        assert_eq!(
            eval_on_df(&pred, df),
            vec![Some(true), Some(false), Some(true)]
        );
    }

    #[test]
    fn test_absolute_date_difference_days() {
        let pred = ComparisonPredicate::AbsoluteDateDifference {
            col: "dob".into(),
            threshold: 30,
            metric: DateMetric::Day,
        };
        // Build Date columns from day offsets via cast from Int32 → Date.
        let df = df!(
            "dob_l" => [0i32, 0, 0],
            "dob_r" => [10i32, 30, 31],
        )
        .unwrap()
        .lazy()
        .with_columns([
            col("dob_l").cast(DataType::Date),
            col("dob_r").cast(DataType::Date),
        ])
        .collect()
        .unwrap();
        // diffs: 10 ≤ 30 true; 30 ≤ 30 true; 31 ≤ 30 false
        assert_eq!(
            eval_on_df(&pred, df),
            vec![Some(true), Some(true), Some(false)]
        );
    }

    #[test]
    fn test_absolute_date_difference_year() {
        let pred = ComparisonPredicate::AbsoluteDateDifference {
            col: "dob".into(),
            threshold: 1,
            metric: DateMetric::Year,
        };
        // 2020-06-15 vs 2021-01-01 → calendar-year diff 1 ≤ 1 → true
        // 2020-06-15 vs 2022-06-15 → 2 > 1 → false
        let df = df!(
            "dob_l" => ["2020-06-15", "2020-06-15"],
            "dob_r" => ["2021-01-01", "2022-06-15"],
        )
        .unwrap()
        .lazy()
        .with_columns([
            col("dob_l").cast(DataType::Date),
            col("dob_r").cast(DataType::Date),
        ])
        .collect()
        .unwrap();
        assert_eq!(eval_on_df(&pred, df), vec![Some(true), Some(false)]);
    }

    #[test]
    fn test_distance_in_km_to_expr() {
        let pred = ComparisonPredicate::DistanceInKm {
            lat_col: "lat".into(),
            long_col: "lon".into(),
            threshold_km: 350.0,
        };
        // London (51.5074, -0.1278) ↔ Paris (48.8566, 2.3522) ≈ 344 km → true at 350
        // London ↔ New York (40.7128, -74.0060) ≈ 5570 km → false
        let df = df!(
            "lat_l" => [51.5074, 51.5074],
            "lon_l" => [-0.1278, -0.1278],
            "lat_r" => [48.8566, 40.7128],
            "lon_r" => [2.3522, -74.0060],
        )
        .unwrap();
        assert_eq!(eval_on_df(&pred, df), vec![Some(true), Some(false)]);
    }

    #[test]
    fn test_array_intersect_to_expr() {
        let pred = ComparisonPredicate::ArrayIntersect {
            col: "tokens".into(),
            min_size: 2,
        };
        let df = df!(
            "tokens_l" => [Series::new("".into(), ["a", "b", "c"]),
                           Series::new("".into(), ["a", "b", "c"])],
            "tokens_r" => [Series::new("".into(), ["b", "c", "d"]),
                           Series::new("".into(), ["x", "y", "z"])],
        )
        .unwrap();
        // {a,b,c}∩{b,c,d}={b,c} size 2 ≥ 2 → true; ∩ with {x,y,z}=∅ → false
        assert_eq!(eval_on_df(&pred, df), vec![Some(true), Some(false)]);
    }

    #[test]
    fn test_cosine_similarity_to_expr() {
        let pred = ComparisonPredicate::CosineSimilarity {
            col: "vec".into(),
            threshold: 0.99,
        };
        let df = df!(
            "vec_l" => [Series::new("".into(), [1.0f64, 0.0, 0.0]),
                        Series::new("".into(), [1.0f64, 0.0, 0.0])],
            "vec_r" => [Series::new("".into(), [1.0f64, 0.0, 0.0]),
                        Series::new("".into(), [0.0f64, 1.0, 0.0])],
        )
        .unwrap();
        // identical → cos 1.0 ≥ 0.99 → true; orthogonal → cos 0.0 → false
        assert_eq!(eval_on_df(&pred, df), vec![Some(true), Some(false)]);
    }

    #[test]
    fn test_and_to_expr() {
        // Exact match AND Jaccard >= 0.5 (both on "name").
        let pred = ComparisonPredicate::And(vec![
            Box::new(ComparisonPredicate::ExactMatch { col: "name".into() }),
            Box::new(ComparisonPredicate::Jaccard {
                col: "name".into(),
                threshold: 0.5,
            }),
        ]);
        // "abc"/"abc": exact ✓ jaccard ✓ → true; "abc"/"abd": exact ✗ → false
        let results = eval_predicate(&pred, &["abc", "abc"], &["abc", "abd"]);
        assert_eq!(results, vec![Some(true), Some(false)]);
    }

    #[test]
    fn test_or_to_expr() {
        // Exact match OR Jaccard >= 0.5.
        let pred = ComparisonPredicate::Or(vec![
            Box::new(ComparisonPredicate::ExactMatch { col: "name".into() }),
            Box::new(ComparisonPredicate::Jaccard {
                col: "name".into(),
                threshold: 0.5,
            }),
        ]);
        // "abc"/"abd": exact ✗ but jaccard 2/4=0.5 ✓ → true;
        // "abc"/"xyz": both ✗ → false
        let results = eval_predicate(&pred, &["abc", "abc"], &["abd", "xyz"]);
        assert_eq!(results, vec![Some(true), Some(false)]);
    }

    #[test]
    fn test_not_to_expr() {
        let pred = ComparisonPredicate::Not(Box::new(ComparisonPredicate::ExactMatch {
            col: "name".into(),
        }));
        // "abc"/"abc": exact → not = false; "abc"/"xyz": not exact → true
        let results = eval_predicate(&pred, &["abc", "abc"], &["abc", "xyz"]);
        assert_eq!(results, vec![Some(false), Some(true)]);
    }

    #[test]
    fn test_custom_predicate_to_expr() {
        let pred = ComparisonPredicate::CustomPredicate {
            dsl: "name_l = name_r".into(),
        };
        let df = df!(
            "name_l" => ["abc", "abc"],
            "name_r" => ["abc", "xyz"],
        )
        .unwrap();
        let expr = pred.to_expr().unwrap();
        let result = df
            .lazy()
            .with_column(expr.alias("result"))
            .collect()
            .unwrap();
        let got: Vec<Option<bool>> = result
            .column("result")
            .unwrap()
            .bool()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(got, vec![Some(true), Some(false)]);
    }

    #[test]
    fn test_custom_predicate_invalid_sql_errors() {
        let pred = ComparisonPredicate::CustomPredicate {
            dsl: "this is not valid sql !!!".into(),
        };
        assert!(matches!(pred.to_expr(), Err(WeldrsError::Config(_))));
    }

    #[test]
    fn test_composition_columns_recurse() {
        let pred = ComparisonPredicate::And(vec![
            Box::new(ComparisonPredicate::ExactMatch {
                col: "first_name".into(),
            }),
            Box::new(ComparisonPredicate::Not(Box::new(
                ComparisonPredicate::ExactMatch { col: "city".into() },
            ))),
        ]);
        let mut cols = pred.columns();
        cols.sort();
        assert_eq!(cols, vec!["city", "first_name"]);
    }

    #[test]
    fn test_predicate_serde_roundtrip_all_variants() {
        let variants = vec![
            ComparisonPredicate::NullCheck { col: "n".into() },
            ComparisonPredicate::ExactMatch { col: "n".into() },
            ComparisonPredicate::LevenshteinDistance {
                col: "n".into(),
                threshold: 2,
            },
            ComparisonPredicate::DamerauLevenshtein {
                col: "n".into(),
                threshold: 1,
            },
            ComparisonPredicate::Hamming {
                col: "n".into(),
                threshold: 1,
            },
            ComparisonPredicate::Jaccard {
                col: "n".into(),
                threshold: 0.7,
            },
            ComparisonPredicate::JaroWinklerSimilarity {
                col: "n".into(),
                threshold: 0.9,
            },
            ComparisonPredicate::And(vec![Box::new(ComparisonPredicate::ExactMatch {
                col: "n".into(),
            })]),
            ComparisonPredicate::Not(Box::new(ComparisonPredicate::ExactMatch {
                col: "n".into(),
            })),
            ComparisonPredicate::CustomPredicate {
                dsl: "n_l = n_r".into(),
            },
            ComparisonPredicate::Else,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: ComparisonPredicate = serde_json::from_str(&json).unwrap();
            // Re-serialize and compare strings (enum has no PartialEq).
            assert_eq!(json, serde_json::to_string(&back).unwrap());
        }
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
