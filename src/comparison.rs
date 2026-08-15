//! Comparison definitions and builder.
//!
//! A [`Comparison`] groups multiple [`ComparisonLevel`]s that target the same
//! input column(s). Use [`ComparisonBuilder`] for ergonomic construction.
//!
//! Comparisons are the core of the Fellegi-Sunter model — each one describes
//! how a particular column (or set of columns) should be compared between two
//! records. Levels are evaluated top-to-bottom; the first matching level
//! determines the gamma value for a given record pair.
//!
//! See [`comparison_level`](crate::comparison_level) for the available
//! predicates and [`settings`](crate::settings) for how comparisons are
//! assembled into a complete model configuration.
//!
//! # Example
//!
//! ```
//! use weldrs::comparison::ComparisonBuilder;
//!
//! let comparison = ComparisonBuilder::new("first_name")
//!     .description("Compare first names with fuzzy matching")
//!     .null_level()
//!     .exact_match_level()
//!     .jaro_winkler_level(0.88)
//!     .else_level()
//!     .build()
//!     .unwrap();
//!
//! assert_eq!(comparison.output_column_name, "first_name");
//! assert_eq!(comparison.comparison_levels.len(), 4);
//! ```

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::comparison_level::{ComparisonLevel, ComparisonPredicate, DateMetric};
use crate::error::Result;
use crate::probability;

fn default_tf_weight() -> f64 {
    1.0
}

/// A comparison defines how a set of input columns are compared to produce a
/// gamma column (comparison vector value) for each record pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comparison {
    /// Name used to derive gamma/BF column names (e.g. `"first_name"`).
    pub output_column_name: String,
    /// Optional human-readable description of this comparison.
    pub description: Option<String>,
    /// The raw column name(s) this comparison operates on.
    pub input_columns: Vec<String>,
    /// Ordered list of levels, evaluated top-to-bottom. The first matching
    /// level determines the gamma value for a given record pair.
    pub comparison_levels: Vec<ComparisonLevel>,
    /// Whether to apply term-frequency adjustments at prediction time. When
    /// enabled, agreeing values are reweighted by their rarity (rare matches
    /// score higher). Requires a single input column with an exact-match level.
    #[serde(default)]
    pub term_frequency_adjustments: bool,
    /// Exponent applied to the term-frequency adjustment (Splink's
    /// `tf_adjustment_weight`). `1.0` = full adjustment, `0.0` = none.
    #[serde(default = "default_tf_weight")]
    pub tf_adjustment_weight: f64,
    /// Floor applied to the relative term frequency before adjustment
    /// (Splink's `tf_minimum_u_value`). `0.0` disables the floor.
    #[serde(default)]
    pub tf_minimum_u_value: f64,
}

impl Comparison {
    /// The gamma column name for this comparison, e.g. `"gamma_first_name"`.
    pub fn gamma_column_name(&self, prefix: &str) -> String {
        format!("{prefix}{}", self.output_column_name)
    }

    /// The Bayes-factor column name, e.g. `"bf_first_name"`.
    pub fn bf_column_name(&self, prefix: &str) -> String {
        format!("{prefix}{}", self.output_column_name)
    }

    /// Non-null comparison levels (those that participate in EM training).
    pub fn non_null_levels(&self) -> Vec<&ComparisonLevel> {
        self.comparison_levels
            .iter()
            .filter(|l| !l.is_null_level)
            .collect()
    }

    /// Non-null comparison levels (mutable).
    pub fn non_null_levels_mut(&mut self) -> Vec<&mut ComparisonLevel> {
        self.comparison_levels
            .iter_mut()
            .filter(|l| !l.is_null_level)
            .collect()
    }

    /// Build a chained `when/then/otherwise` expression that evaluates each
    /// comparison level in order, returning the comparison vector value.
    pub fn gamma_expr(&self, prefix: &str) -> Result<Expr> {
        let col_name = self.gamma_column_name(prefix);

        // Build from last to first: the final `otherwise` is always -1 (should
        // never be reached if levels are exhaustive, but acts as a safeguard).
        let mut expr: Option<Expr> = None;

        // Iterate in reverse so that we build the chain from the inside out.
        for level in self.comparison_levels.iter().rev() {
            match &level.predicate {
                ComparisonPredicate::Else => {
                    // The Else level is the innermost `otherwise`.
                    expr = Some(lit(level.comparison_vector_value));
                }
                predicate => {
                    let condition = predicate.to_expr()?;
                    let inner = expr.unwrap_or(lit(-1i32));
                    expr = Some(
                        when(condition)
                            .then(lit(level.comparison_vector_value))
                            .otherwise(inner),
                    );
                }
            }
        }

        Ok(expr
            .unwrap_or(lit(-1i32))
            .cast(DataType::Int8)
            .alias(col_name.as_str()))
    }

    /// Build an expression that maps gamma values to Bayes factors.
    ///
    /// For null levels the BF is 1.0 (neutral).
    pub fn bf_expr(&self, gamma_prefix: &str, bf_prefix: &str) -> Result<Expr> {
        let gamma_col = self.gamma_column_name(gamma_prefix);
        let bf_col = self.bf_column_name(bf_prefix);

        let mut expr: Expr = lit(1.0); // default BF for anything unmatched

        for level in &self.comparison_levels {
            let bf = if level.is_null_level {
                1.0
            } else {
                level.bayes_factor().unwrap_or(1.0)
            };

            expr = when(col(gamma_col.as_str()).eq(lit(level.comparison_vector_value)))
                .then(lit(bf))
                .otherwise(expr);
        }

        Ok(expr.alias(bf_col.as_str()))
    }

    /// Build the per-row term-frequency adjustment multiplier for this
    /// comparison, or `None` if term-frequency adjustments are disabled.
    ///
    /// For every non-null, non-`Else` level the multiplier is
    /// `(u_exact / max(tf_l, tf_r))^weight`, where `u_exact` is the
    /// u-probability of the exact-match level and `tf_l` / `tf_r` are the
    /// relative term frequencies of the two values (read from the suffixed
    /// `{col}_tf_l` / `{col}_tf_r` columns). The null and catch-all levels get
    /// a neutral multiplier of `1.0`. Null term frequencies also fall back to
    /// `1.0`.
    ///
    /// Returns `None` if the comparison has no single input column, no
    /// exact-match level, or an untrained exact-match u-probability.
    pub fn tf_adjustment_expr(&self, gamma_prefix: &str) -> Option<Expr> {
        if !self.term_frequency_adjustments {
            return None;
        }
        let tf_col = self.input_columns.first()?;
        let u_exact = self
            .comparison_levels
            .iter()
            .find(|l| matches!(l.predicate, ComparisonPredicate::ExactMatch { .. }))
            .and_then(|l| l.u_probability)?;

        let gamma_col = self.gamma_column_name(gamma_prefix);
        let tf_l = col(format!("{tf_col}_tf_l"));
        let tf_r = col(format!("{tf_col}_tf_r"));

        // max(tf_l, tf_r), floored by tf_minimum_u_value.
        let max_tf = when(tf_l.clone().gt_eq(tf_r.clone()))
            .then(tf_l)
            .otherwise(tf_r);
        let floor = self.tf_minimum_u_value;
        let max_tf = when(max_tf.clone().lt(lit(floor)))
            .then(lit(floor))
            .otherwise(max_tf);

        let tf_term = (lit(u_exact) / max_tf).pow(lit(self.tf_adjustment_weight));

        // Apply the multiplier to every non-null, non-Else level.
        let mut expr: Expr = lit(1.0);
        for level in &self.comparison_levels {
            if level.is_null_level || matches!(level.predicate, ComparisonPredicate::Else) {
                continue;
            }
            expr = when(col(gamma_col.as_str()).eq(lit(level.comparison_vector_value)))
                .then(tf_term.clone())
                .otherwise(expr);
        }

        // Null term frequencies (or any null arithmetic) → neutral 1.0.
        Some(expr.fill_null(lit(1.0)))
    }

    /// The term-frequency adjustment column name, e.g. `"tf_first_name"`.
    pub fn tf_column_name(&self, prefix: &str) -> String {
        format!("{prefix}{}", self.output_column_name)
    }

    /// Whether all non-null levels have trained m-probabilities.
    pub fn m_is_trained(&self) -> bool {
        self.non_null_levels()
            .iter()
            .all(|l| l.m_probability.is_some())
    }

    /// Whether all non-null levels have trained u-probabilities.
    pub fn u_is_trained(&self) -> bool {
        self.non_null_levels()
            .iter()
            .all(|l| l.u_probability.is_some())
    }
}

/// Builder for constructing a [`Comparison`] with ergonomic chaining.
pub struct ComparisonBuilder {
    output_column_name: String,
    description: Option<String>,
    levels: Vec<(ComparisonPredicate, String, bool)>, // (predicate, label, is_null)
    term_frequency_adjustments: bool,
    tf_adjustment_weight: f64,
    tf_minimum_u_value: f64,
}

impl ComparisonBuilder {
    /// Start building a comparison for the given column.
    ///
    /// # Examples
    ///
    /// ```
    /// use weldrs::comparison::ComparisonBuilder;
    ///
    /// let comparison = ComparisonBuilder::new("first_name")
    ///     .null_level()
    ///     .exact_match_level()
    ///     .jaro_winkler_level(0.88)
    ///     .else_level()
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(comparison.comparison_levels.len(), 4);
    /// ```
    pub fn new(column: &str) -> Self {
        Self {
            output_column_name: column.to_string(),
            description: None,
            levels: Vec::new(),
            term_frequency_adjustments: false,
            tf_adjustment_weight: 1.0,
            tf_minimum_u_value: 0.0,
        }
    }

    /// Enable term-frequency adjustments for this comparison (rare agreeing
    /// values score higher). Requires a single input column and an
    /// exact-match level; validated in [`build`](Self::build).
    pub fn with_term_frequency_adjustments(mut self) -> Self {
        self.term_frequency_adjustments = true;
        self
    }

    /// Set the term-frequency adjustment exponent (default `1.0`).
    pub fn tf_adjustment_weight(mut self, weight: f64) -> Self {
        self.tf_adjustment_weight = weight;
        self
    }

    /// Set the floor applied to relative term frequency before adjustment
    /// (default `0.0`, i.e. no floor).
    pub fn tf_minimum_u_value(mut self, min_u: f64) -> Self {
        self.tf_minimum_u_value = min_u;
        self
    }

    /// Set an optional human-readable description for this comparison.
    pub fn description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Add a null-check level (both values are null).
    pub fn null_level(mut self) -> Self {
        let col = self.output_column_name.clone();
        self.levels.push((
            ComparisonPredicate::NullCheck { col },
            "Null".to_string(),
            true,
        ));
        self
    }

    /// Add an exact-match level.
    pub fn exact_match_level(mut self) -> Self {
        let col = self.output_column_name.clone();
        self.levels.push((
            ComparisonPredicate::ExactMatch { col },
            "Exact match".to_string(),
            false,
        ));
        self
    }

    /// Add a Levenshtein distance level.
    pub fn levenshtein_level(mut self, threshold: u32) -> Self {
        let col = self.output_column_name.clone();
        self.levels.push((
            ComparisonPredicate::LevenshteinDistance { col, threshold },
            format!("Levenshtein <= {threshold}"),
            false,
        ));
        self
    }

    /// Add a Jaro-Winkler similarity level.
    pub fn jaro_winkler_level(mut self, threshold: f64) -> Self {
        let col = self.output_column_name.clone();
        self.levels.push((
            ComparisonPredicate::JaroWinklerSimilarity { col, threshold },
            format!("Jaro-Winkler >= {threshold}"),
            false,
        ));
        self
    }

    /// Add a Jaro similarity level.
    pub fn jaro_level(mut self, threshold: f64) -> Self {
        let col = self.output_column_name.clone();
        self.levels.push((
            ComparisonPredicate::JaroSimilarity { col, threshold },
            format!("Jaro >= {threshold}"),
            false,
        ));
        self
    }

    /// Add a Damerau-Levenshtein distance level (Levenshtein plus adjacent
    /// transpositions).
    pub fn damerau_levenshtein_level(mut self, threshold: u32) -> Self {
        let col = self.output_column_name.clone();
        self.levels.push((
            ComparisonPredicate::DamerauLevenshtein { col, threshold },
            format!("Damerau-Levenshtein <= {threshold}"),
            false,
        ));
        self
    }

    /// Add a Hamming distance level (differing positions; equal-length values
    /// only).
    pub fn hamming_level(mut self, threshold: u32) -> Self {
        let col = self.output_column_name.clone();
        self.levels.push((
            ComparisonPredicate::Hamming { col, threshold },
            format!("Hamming <= {threshold}"),
            false,
        ));
        self
    }

    /// Add a Jaccard similarity level (character-set Jaccard).
    pub fn jaccard_level(mut self, threshold: f64) -> Self {
        let col = self.output_column_name.clone();
        self.levels.push((
            ComparisonPredicate::Jaccard { col, threshold },
            format!("Jaccard >= {threshold}"),
            false,
        ));
        self
    }

    /// Add an absolute date-difference level (column must be a Date dtype).
    pub fn date_difference_level(mut self, threshold: i64, metric: DateMetric) -> Self {
        let col = self.output_column_name.clone();
        self.levels.push((
            ComparisonPredicate::AbsoluteDateDifference {
                col,
                threshold,
                metric,
            },
            format!("Date difference <= {threshold} {metric:?}"),
            false,
        ));
        self
    }

    /// Add a percentage-difference level for a numeric column.
    pub fn percentage_difference_level(mut self, threshold: f64) -> Self {
        let col = self.output_column_name.clone();
        self.levels.push((
            ComparisonPredicate::PercentageDifference { col, threshold },
            format!("Percentage difference <= {threshold}"),
            false,
        ));
        self
    }

    /// Add a great-circle distance level over latitude/longitude columns
    /// (degrees), matching within `threshold_km` kilometres.
    pub fn distance_km_level(mut self, lat_col: &str, long_col: &str, threshold_km: f64) -> Self {
        self.levels.push((
            ComparisonPredicate::DistanceInKm {
                lat_col: lat_col.to_string(),
                long_col: long_col.to_string(),
                threshold_km,
            },
            format!("Distance <= {threshold_km} km"),
            false,
        ));
        self
    }

    /// Add an array-intersection level (list columns sharing at least
    /// `min_size` elements).
    pub fn array_intersect_level(mut self, min_size: u32) -> Self {
        let col = self.output_column_name.clone();
        self.levels.push((
            ComparisonPredicate::ArrayIntersect { col, min_size },
            format!("Array intersection >= {min_size}"),
            false,
        ));
        self
    }

    /// Add a cosine-similarity level for a numeric list (vector) column.
    pub fn cosine_level(mut self, threshold: f64) -> Self {
        let col = self.output_column_name.clone();
        self.levels.push((
            ComparisonPredicate::CosineSimilarity { col, threshold },
            format!("Cosine similarity >= {threshold}"),
            false,
        ));
        self
    }

    /// Add a level that holds when **all** of the given predicates hold.
    ///
    /// Use [`ComparisonPredicate`] variants to compose the conditions, e.g. an
    /// exact match on one column AND a fuzzy match on another.
    pub fn and_level(mut self, label: &str, predicates: Vec<ComparisonPredicate>) -> Self {
        let boxed = predicates.into_iter().map(Box::new).collect();
        self.levels
            .push((ComparisonPredicate::And(boxed), label.to_string(), false));
        self
    }

    /// Add a level that holds when **any** of the given predicates hold.
    pub fn or_level(mut self, label: &str, predicates: Vec<ComparisonPredicate>) -> Self {
        let boxed = predicates.into_iter().map(Box::new).collect();
        self.levels
            .push((ComparisonPredicate::Or(boxed), label.to_string(), false));
        self
    }

    /// Add a level that holds when the given predicate does **not** hold.
    pub fn not_level(mut self, label: &str, predicate: ComparisonPredicate) -> Self {
        self.levels.push((
            ComparisonPredicate::Not(Box::new(predicate)),
            label.to_string(),
            false,
        ));
        self
    }

    /// Add a level defined by a raw Polars-SQL boolean condition over the
    /// suffixed `{col}_l` / `{col}_r` columns (the Rust analog of Splink's
    /// `CustomLevel`).
    ///
    /// Note: columns referenced only inside a custom condition are **not**
    /// auto-tracked in the comparison's `input_columns`. Ensure those columns
    /// are also referenced by another level in the same comparison, or are
    /// otherwise present in the frame being scored.
    pub fn custom_level(mut self, label: &str, dsl: &str) -> Self {
        self.levels.push((
            ComparisonPredicate::CustomPredicate {
                dsl: dsl.to_string(),
            },
            label.to_string(),
            false,
        ));
        self
    }

    /// Add the catch-all "else" level.
    pub fn else_level(mut self) -> Self {
        self.levels.push((
            ComparisonPredicate::Else,
            "All other comparisons".to_string(),
            false,
        ));
        self
    }

    /// Finalise the comparison. Assigns comparison vector values automatically:
    /// - Null level → -1
    /// - Highest match level → N-1 (where N = number of non-null levels)
    /// - Else → 0
    ///
    /// # Errors
    ///
    /// Returns [`WeldrsError::Config`] if:
    /// - No levels have been added.
    /// - There is no `else_level` (the catch-all is required to make the
    ///   comparison exhaustive).
    /// - The `else_level` is not the last non-null level (it must be added
    ///   last so that higher-priority predicates are evaluated first).
    /// - There are no non-null comparison levels (at least one is required
    ///   for EM training to be meaningful).
    pub fn build(self) -> Result<Comparison> {
        use crate::error::WeldrsError;

        // ── Validation ───────────────────────────────────────────────
        if self.levels.is_empty() {
            return Err(WeldrsError::Config(
                "ComparisonBuilder: at least one level is required".into(),
            ));
        }

        let has_else = self
            .levels
            .iter()
            .any(|(p, _, _)| matches!(p, ComparisonPredicate::Else));
        if !has_else {
            return Err(WeldrsError::Config(format!(
                "ComparisonBuilder for '{}': an else_level is required as a catch-all",
                self.output_column_name
            )));
        }

        // Else must be the last non-null level.
        let last_non_null = self.levels.iter().rev().find(|(_, _, is_null)| !is_null);
        if let Some((pred, _, _)) = last_non_null
            && !matches!(pred, ComparisonPredicate::Else)
        {
            return Err(WeldrsError::Config(format!(
                "ComparisonBuilder for '{}': else_level must be the last level added \
                 (non-null levels after else would be unreachable)",
                self.output_column_name
            )));
        }

        let non_null_count = self.levels.iter().filter(|(_, _, null)| !null).count();
        if non_null_count == 0 {
            return Err(WeldrsError::Config(format!(
                "ComparisonBuilder for '{}': at least one non-null level is required",
                self.output_column_name
            )));
        }

        // ── Build ────────────────────────────────────────────────────

        // Assign default m/u probabilities for non-null levels.
        let m_defaults = probability::default_m_values(non_null_count);
        let u_defaults = probability::default_u_values(non_null_count);

        let mut non_null_index = 0;
        let mut comparison_levels = Vec::with_capacity(self.levels.len());

        // Non-null levels are numbered in descending order: highest match
        // level gets the largest value.  The Else level always gets 0.
        // Other non-null levels are numbered from (non_null_count - 1) down to 0,
        // but Else is always the last non-null level added and gets 0.
        //
        // More precisely, we assign comparison_vector_value as:
        //   null → -1
        //   first non-null (highest quality match) → non_null_count - 1
        //   second non-null → non_null_count - 2
        //   ...
        //   else → 0
        for (predicate, label, is_null) in self.levels {
            let cv_value = if is_null {
                -1
            } else {
                let v = (non_null_count - 1 - non_null_index) as i32;
                non_null_index += 1;
                v
            };

            let (m_prob, u_prob) = if is_null {
                (None, None)
            } else {
                // Map from descending cv index to the default arrays which are
                // ordered ascending (index 0 = else/lowest, last = highest).
                let array_idx = cv_value as usize;
                (Some(m_defaults[array_idx]), Some(u_defaults[array_idx]))
            };

            comparison_levels.push(ComparisonLevel {
                predicate,
                label,
                is_null_level: is_null,
                comparison_vector_value: cv_value,
                m_probability: m_prob,
                u_probability: u_prob,
                fix_m_probability: false,
                fix_u_probability: false,
            });
        }

        // Collect input columns deterministically (sorted) so that
        // downstream code that iterates over them produces stable results.
        let mut input_columns: Vec<String> = comparison_levels
            .iter()
            .flat_map(|l| l.predicate.columns())
            .map(String::from)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        input_columns.sort();

        // Term-frequency adjustments require a single input column and an
        // exact-match level on it (the exact-match u-probability is the
        // adjustment baseline).
        if self.term_frequency_adjustments {
            if input_columns.len() != 1 {
                return Err(WeldrsError::Config(format!(
                    "ComparisonBuilder for '{}': term-frequency adjustments require exactly one \
                     input column, found {}",
                    self.output_column_name,
                    input_columns.len()
                )));
            }
            let has_exact = comparison_levels
                .iter()
                .any(|l| matches!(l.predicate, ComparisonPredicate::ExactMatch { .. }));
            if !has_exact {
                return Err(WeldrsError::Config(format!(
                    "ComparisonBuilder for '{}': term-frequency adjustments require an \
                     exact-match level on the input column",
                    self.output_column_name
                )));
            }
        }

        Ok(Comparison {
            output_column_name: self.output_column_name,
            description: self.description,
            input_columns,
            comparison_levels,
            term_frequency_adjustments: self.term_frequency_adjustments,
            tf_adjustment_weight: self.tf_adjustment_weight,
            tf_minimum_u_value: self.tf_minimum_u_value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers;

    #[test]
    fn test_builder_two_level_cv_values() {
        // null + exact + else → 3 levels
        let comp = test_helpers::exact_match_comparison("name");
        let cvs: Vec<i32> = comp
            .comparison_levels
            .iter()
            .map(|l| l.comparison_vector_value)
            .collect();
        // null=-1, exact=1, else=0
        assert_eq!(cvs, vec![-1, 1, 0]);
    }

    #[test]
    fn test_builder_three_level_cv_values() {
        // null + exact + jaro_winkler + else → 4 levels, 3 non-null
        let comp = test_helpers::fuzzy_comparison("name", 0.85);
        let cvs: Vec<i32> = comp
            .comparison_levels
            .iter()
            .map(|l| l.comparison_vector_value)
            .collect();
        // null=-1, exact=2, jw=1, else=0
        assert_eq!(cvs, vec![-1, 2, 1, 0]);
    }

    #[test]
    fn test_builder_default_m_assignment() {
        let comp = test_helpers::exact_match_comparison("name");
        // Null level has None m
        assert!(comp.comparison_levels[0].m_probability.is_none());
        // Non-null levels have Some(m)
        for level in comp.non_null_levels() {
            assert!(level.m_probability.is_some());
        }
    }

    #[test]
    fn test_builder_default_u_assignment() {
        let comp = test_helpers::exact_match_comparison("name");
        // Null level has None u
        assert!(comp.comparison_levels[0].u_probability.is_none());
        // Non-null levels have Some(u)
        for level in comp.non_null_levels() {
            assert!(level.u_probability.is_some());
        }
    }

    #[test]
    fn test_builder_input_columns() {
        let comp = test_helpers::exact_match_comparison("first_name");
        assert_eq!(comp.input_columns.len(), 1);
        assert!(comp.input_columns.contains(&"first_name".to_string()));
    }

    #[test]
    fn test_gamma_expr_exact_match() {
        let comp = test_helpers::exact_match_comparison("name");
        let df = test_helpers::make_paired_df(
            &[1, 2, 3],
            &[4, 5, 6],
            "name",
            &["Alice", "Bob", "Charlie"],
            &["Alice", "Carol", "Charlie"],
        );
        let gamma_expr = comp.gamma_expr("gamma_").unwrap();
        let result = df.lazy().with_column(gamma_expr).collect().unwrap();
        let gammas: Vec<Option<i8>> = result
            .column("gamma_name")
            .unwrap()
            .i8()
            .unwrap()
            .into_iter()
            .collect();
        // Alice==Alice → exact(1), Bob!=Carol → else(0), Charlie==Charlie → exact(1)
        assert_eq!(gammas, vec![Some(1i8), Some(0i8), Some(1i8)]);
    }

    #[test]
    fn test_gamma_expr_fuzzy_levels() {
        let comp = test_helpers::fuzzy_comparison("name", 0.85);
        let df = test_helpers::make_paired_df(
            &[1, 2, 3],
            &[4, 5, 6],
            "name",
            &["martha", "abc", "exact"],
            &["marhta", "xyz", "exact"],
        );
        let gamma_expr = comp.gamma_expr("gamma_").unwrap();
        let result = df.lazy().with_column(gamma_expr).collect().unwrap();
        let gammas: Vec<Option<i8>> = result
            .column("gamma_name")
            .unwrap()
            .i8()
            .unwrap()
            .into_iter()
            .collect();
        // "martha"/"marhta" → JW ~0.96 → jw level (cv=1)
        // "abc"/"xyz" → low JW → else (cv=0)
        // "exact"/"exact" → exact match (cv=2)
        assert_eq!(gammas, vec![Some(1i8), Some(0i8), Some(2i8)]);
    }

    #[test]
    fn test_bf_expr_mapping() {
        let mut comp = test_helpers::exact_match_comparison("name");
        // Set known m/u values for non-null levels
        for level in &mut comp.comparison_levels {
            if level.is_null_level {
                continue;
            }
            if level.comparison_vector_value == 1 {
                // exact match
                level.m_probability = Some(0.9);
                level.u_probability = Some(0.1);
            } else {
                // else
                level.m_probability = Some(0.1);
                level.u_probability = Some(0.9);
            }
        }

        // Build a DF with gamma values
        let df = df!(
            "gamma_name" => [1i8, 0i8, -1i8],
        )
        .unwrap();

        let bf_expr = comp.bf_expr("gamma_", "bf_").unwrap();
        let result = df.lazy().with_column(bf_expr).collect().unwrap();
        let bfs: Vec<Option<f64>> = result
            .column("bf_name")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .collect();

        // exact match: m=0.9/u=0.1 = 9.0
        assert!((bfs[0].unwrap() - 9.0).abs() < 1e-10);
        // else: m=0.1/u=0.9 ≈ 0.111
        assert!((bfs[1].unwrap() - 0.1 / 0.9).abs() < 1e-10);
        // null: BF = 1.0
        assert!((bfs[2].unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_builder_rejects_empty() {
        let result = ComparisonBuilder::new("name").build();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("at least one level"), "got: {msg}");
    }

    #[test]
    fn test_builder_rejects_missing_else() {
        let result = ComparisonBuilder::new("name")
            .null_level()
            .exact_match_level()
            .build();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("else_level is required"), "got: {msg}");
    }

    #[test]
    fn test_builder_rejects_else_not_last() {
        let result = ComparisonBuilder::new("name")
            .null_level()
            .else_level()
            .exact_match_level()
            .build();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("else_level must be the last"), "got: {msg}");
    }

    #[test]
    fn test_builder_rejects_only_null_levels() {
        let result = ComparisonBuilder::new("name").null_level().build();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        // Hits both "else_level is required" and conceptually "no non-null levels"
        assert!(msg.contains("else_level is required"), "got: {msg}");
    }

    #[test]
    fn test_builder_input_columns_sorted() {
        // Even if internal ordering is nondeterministic, output should be sorted.
        let comp = ComparisonBuilder::new("name")
            .null_level()
            .exact_match_level()
            .else_level()
            .build()
            .unwrap();
        let cols = &comp.input_columns;
        let mut sorted = cols.clone();
        sorted.sort();
        assert_eq!(cols, &sorted);
    }

    #[test]
    fn test_builder_composition_and_custom_levels_gamma() {
        use crate::comparison_level::ComparisonPredicate;
        // null / (exact AND jaccard>=0.5) / custom(name_l = name_r reversed-ish) / else
        let comp = ComparisonBuilder::new("name")
            .null_level()
            .and_level(
                "Exact & Jaccard",
                vec![
                    ComparisonPredicate::ExactMatch { col: "name".into() },
                    ComparisonPredicate::Jaccard {
                        col: "name".into(),
                        threshold: 0.5,
                    },
                ],
            )
            .custom_level("Custom equality", "name_l = name_r")
            .else_level()
            .build()
            .unwrap();

        // 3 non-null levels: and=2, custom=1, else=0; null=-1
        let cvs: Vec<i32> = comp
            .comparison_levels
            .iter()
            .map(|l| l.comparison_vector_value)
            .collect();
        assert_eq!(cvs, vec![-1, 2, 1, 0]);

        let df = test_helpers::make_paired_df(
            &[1, 2, 3],
            &[4, 5, 6],
            "name",
            &["Alice", "Bob", "Zoe"],
            &["Alice", "Bob", "Xan"],
        );
        let result = df
            .lazy()
            .with_column(comp.gamma_expr("gamma_").unwrap())
            .collect()
            .unwrap();
        let gammas: Vec<Option<i8>> = result
            .column("gamma_name")
            .unwrap()
            .i8()
            .unwrap()
            .into_iter()
            .collect();
        // Alice==Alice → and-level (2); Bob==Bob → and-level (2); Zoe!=Xan → else (0)
        assert_eq!(gammas, vec![Some(2i8), Some(2i8), Some(0i8)]);
    }

    #[test]
    fn test_m_u_is_trained() {
        let comp = test_helpers::exact_match_comparison("name");
        // Default build assigns m and u to non-null levels
        assert!(comp.m_is_trained());
        assert!(comp.u_is_trained());

        // Remove m from one level
        let mut comp2 = comp.clone();
        comp2.comparison_levels[1].m_probability = None;
        assert!(!comp2.m_is_trained());
        assert!(comp2.u_is_trained());
    }
}
