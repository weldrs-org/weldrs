//! Comparison definitions and builder.
//!
//! A [`Comparison`] groups multiple [`ComparisonLevel`]s that target the same
//! input column(s). Use [`ComparisonBuilder`] for ergonomic construction.

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::comparison_level::{ComparisonLevel, ComparisonPredicate};
use crate::error::Result;
use crate::probability;

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

        Ok(expr.unwrap_or(lit(-1i32)).alias(col_name.as_str()))
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
}

impl ComparisonBuilder {
    /// Start building a comparison for the given column.
    pub fn new(column: &str) -> Self {
        Self {
            output_column_name: column.to_string(),
            description: None,
            levels: Vec::new(),
        }
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
    pub fn build(self) -> Comparison {
        let non_null_count = self.levels.iter().filter(|(_, _, null)| !null).count();

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

        let input_columns = comparison_levels
            .iter()
            .filter_map(|l| l.predicate.column().map(String::from))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        Comparison {
            output_column_name: self.output_column_name,
            description: self.description,
            input_columns,
            comparison_levels,
        }
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
        let gammas: Vec<Option<i32>> = result
            .column("gamma_name")
            .unwrap()
            .i32()
            .unwrap()
            .into_iter()
            .collect();
        // Alice==Alice → exact(1), Bob!=Carol → else(0), Charlie==Charlie → exact(1)
        assert_eq!(gammas, vec![Some(1), Some(0), Some(1)]);
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
        let gammas: Vec<Option<i32>> = result
            .column("gamma_name")
            .unwrap()
            .i32()
            .unwrap()
            .into_iter()
            .collect();
        // "martha"/"marhta" → JW ~0.96 → jw level (cv=1)
        // "abc"/"xyz" → low JW → else (cv=0)
        // "exact"/"exact" → exact match (cv=2)
        assert_eq!(gammas, vec![Some(1), Some(0), Some(2)]);
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
            "gamma_name" => [1i32, 0, -1],
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
