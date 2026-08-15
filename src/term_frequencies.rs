//! Term-frequency (TF) tables and attachment.
//!
//! Term-frequency adjustments reweight agreeing values by their rarity: a match
//! on a rare value (e.g. surname "Zelenskyy") is stronger evidence than a match
//! on a common one (e.g. "Smith"). At prediction time the Bayes factor of an
//! agreeing level is multiplied by an adjustment derived from the relative term
//! frequency of the value — see
//! [`Comparison::tf_adjustment_expr`](crate::comparison::Comparison::tf_adjustment_expr).
//!
//! This module computes per-column relative term frequencies and attaches them
//! to the input frame *before* blocking, so the existing `_l` / `_r` suffixing
//! produces `{col}_tf_l` / `{col}_tf_r` columns automatically.
//!
//! TF affects only prediction and explanation — EM training and u-estimation
//! operate on raw agreement patterns and never see term frequencies.

use polars::prelude::*;

use crate::comparison::Comparison;
use crate::error::{Result, WeldrsError};

/// The suffix used for a column's relative-term-frequency column.
fn tf_value_column(col_name: &str) -> String {
    format!("{col_name}_tf")
}

/// Compute the relative term-frequency table for a single column.
///
/// Returns a frame with two columns: the original `col` values and
/// `{col}_tf`, where `tf = count(value) / count(non_null_rows)` (matching
/// Splink's term-frequency definition). Null values are excluded.
///
/// # Errors
///
/// Returns an error if the Polars group-by / aggregation fails.
pub fn compute_tf_table(lf: &LazyFrame, col_name: &str) -> Result<DataFrame> {
    let tf_name = tf_value_column(col_name);
    lf.clone()
        .filter(col(col_name).is_not_null())
        .group_by([col(col_name)])
        .agg([len().alias("__tf_count")])
        .with_column(
            (col("__tf_count").cast(DataType::Float64)
                / col("__tf_count").sum().cast(DataType::Float64))
            .alias(tf_name.as_str()),
        )
        .select([col(col_name), col(tf_name.as_str())])
        .collect()
        .map_err(|e| WeldrsError::Training {
            stage: "term_frequencies",
            message: format!("Failed to compute TF table for '{col_name}': {e}"),
        })
}

/// Attach relative term-frequency columns to `lf` for every comparison that has
/// term-frequency adjustments enabled.
///
/// For each such comparison's input column `c`, a `{c}_tf` column is left-joined
/// onto the frame. If no comparison uses term frequencies, `lf` is returned
/// unchanged (zero overhead). Columns are deduplicated so two comparisons over
/// the same column only join once.
///
/// # Errors
///
/// Returns an error if a TF-enabled comparison lacks a single input column, or
/// if a Polars join fails.
pub fn attach_tf_columns(lf: LazyFrame, comparisons: &[Comparison]) -> Result<LazyFrame> {
    let source = lf.clone();
    attach_tf_columns_from(lf, &source, comparisons)
}

/// Attach term-frequency columns to `target`, computing the frequencies from a
/// separate `tf_source` frame.
///
/// Used when scoring records against a reference population (e.g.
/// [`Linker::find_matches_to_new_records`](crate::linker::Linker::find_matches_to_new_records)),
/// where term frequencies must come from the existing dataset rather than the
/// new records being scored.
///
/// # Errors
///
/// Returns an error if a TF-enabled comparison lacks a single input column, or
/// if a Polars join fails.
pub fn attach_tf_columns_from(
    target: LazyFrame,
    tf_source: &LazyFrame,
    comparisons: &[Comparison],
) -> Result<LazyFrame> {
    let mut out = target;
    let mut attached: Vec<String> = Vec::new();

    for comp in comparisons.iter().filter(|c| c.term_frequency_adjustments) {
        let col_name = comp.input_columns.first().ok_or_else(|| WeldrsError::Config(
            format!(
                "Comparison '{}' has term-frequency adjustments enabled but no input column",
                comp.output_column_name
            ),
        ))?;
        if attached.iter().any(|c| c == col_name) {
            continue;
        }

        let tf_table = compute_tf_table(tf_source, col_name)?;
        out = out.join(
            tf_table.lazy(),
            [col(col_name.as_str())],
            [col(col_name.as_str())],
            JoinArgs::new(JoinType::Left),
        );
        attached.push(col_name.clone());
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_lf() -> LazyFrame {
        df!(
            "unique_id" => [1i64, 2, 3, 4, 5],
            // "Smith" x3 (common), "Jones" x1, "Zhang" x1 (rare)
            "surname" => ["Smith", "Smith", "Smith", "Jones", "Zhang"],
        )
        .unwrap()
        .lazy()
    }

    #[test]
    fn test_compute_tf_table_relative_frequencies() {
        let tf = compute_tf_table(&sample_lf(), "surname").unwrap();
        let names: Vec<&str> = tf
            .get_column_names()
            .into_iter()
            .map(|s| s.as_str())
            .collect();
        assert!(names.contains(&"surname"));
        assert!(names.contains(&"surname_tf"));

        // Build a value→tf map.
        let surnames = tf.column("surname").unwrap().str().unwrap();
        let tfs = tf.column("surname_tf").unwrap().f64().unwrap();
        let mut map = std::collections::HashMap::new();
        for (s, t) in surnames.into_iter().zip(tfs) {
            map.insert(s.unwrap().to_string(), t.unwrap());
        }
        // 3/5, 1/5, 1/5
        assert!((map["Smith"] - 0.6).abs() < 1e-12);
        assert!((map["Jones"] - 0.2).abs() < 1e-12);
        assert!((map["Zhang"] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_attach_tf_columns_noop_without_tf() {
        let comps = vec![crate::test_helpers::exact_match_comparison("surname")];
        let out = attach_tf_columns(sample_lf(), &comps)
            .unwrap()
            .collect()
            .unwrap();
        // No TF comparison → no extra columns.
        assert!(!out
            .get_column_names()
            .iter()
            .any(|n| n.as_str() == "surname_tf"));
    }

    #[test]
    fn test_attach_tf_columns_adds_tf_column() {
        let comp = crate::comparison::ComparisonBuilder::new("surname")
            .null_level()
            .exact_match_level()
            .else_level()
            .with_term_frequency_adjustments()
            .build()
            .unwrap();
        let out = attach_tf_columns(sample_lf(), &[comp])
            .unwrap()
            .collect()
            .unwrap();
        assert!(out
            .get_column_names()
            .iter()
            .any(|n| n.as_str() == "surname_tf"));
        assert_eq!(out.height(), 5);
    }
}
