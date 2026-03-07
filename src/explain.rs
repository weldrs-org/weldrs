//! Model explanation and waterfall charts.
//!
//! Provides structured breakdowns of how each comparison contributed to a
//! record pair's match score, plus a summary view of the entire trained model.
//!
//! After [`predict`](crate::predict) scores candidate pairs, use
//! [`explain_pair`] to produce a step-by-step [`WaterfallChart`] showing
//! the prior and each comparison's Bayes factor contribution. Use
//! [`model_summary`] for a high-level [`ModelSummary`] of the trained
//! parameters.
//!
//! For SVG rendering of these structures, see
//! [`visualize`](crate::visualize) (requires the `visualize` feature).

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::comparison::Comparison;
use crate::error::{Result, WeldrsError};
use crate::probability;
use crate::settings::Settings;

// ── Data structures ──────────────────────────────────────────────────

/// A single step in the waterfall: either the prior or one comparison's contribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterfallStep {
    /// Name of the comparison column, or `"Prior"` for the prior step.
    pub column_name: String,
    /// Human-readable label for the matched level (e.g. `"Exact match"`).
    pub label: String,
    /// The gamma value for this comparison (`None` for the prior step).
    pub comparison_vector_value: Option<i32>,
    /// Left record's value for the comparison column.
    pub value_l: Option<String>,
    /// Right record's value for the comparison column.
    pub value_r: Option<String>,
    /// Bayes factor (m/u) contributed by this step.
    pub bayes_factor: f64,
    /// Log2 of the Bayes factor (the match weight contributed by this step).
    pub log2_bayes_factor: f64,
    /// The m-probability of the matched level (`None` for prior or null levels).
    pub m_probability: Option<f64>,
    /// The u-probability of the matched level (`None` for prior or null levels).
    pub u_probability: Option<f64>,
    /// Running total of match weight after this step.
    pub cumulative_match_weight: f64,
    /// Running match probability after this step.
    pub cumulative_match_probability: f64,
}

/// Complete waterfall for one record pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterfallChart {
    /// Unique identifier of the left record.
    pub unique_id_l: String,
    /// Unique identifier of the right record.
    pub unique_id_r: String,
    /// Ordered list of waterfall steps (prior, then one per comparison).
    pub steps: Vec<WaterfallStep>,
    /// Final combined match weight for this pair.
    pub final_match_weight: f64,
    /// Final match probability for this pair.
    pub final_match_probability: f64,
}

/// Summary of a single comparison level's trained parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelSummary {
    /// Human-readable label (e.g. `"Exact match"`).
    pub label: String,
    /// The gamma value assigned to this level.
    pub comparison_vector_value: i32,
    /// Whether this is the null-handling level.
    pub is_null_level: bool,
    /// Trained m-probability (`None` for null levels).
    pub m_probability: Option<f64>,
    /// Trained u-probability (`None` for null levels).
    pub u_probability: Option<f64>,
    /// Bayes factor m/u (`None` for null levels).
    pub bayes_factor: Option<f64>,
    /// Log2 of the Bayes factor (`None` for null levels).
    pub log2_bayes_factor: Option<f64>,
}

/// Summary of a single comparison's trained parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    /// Column name this comparison targets.
    pub output_column_name: String,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Raw input column name(s).
    pub input_columns: Vec<String>,
    /// Level summaries, sorted by match quality descending (null last).
    pub levels: Vec<LevelSummary>,
}

/// Summary of the entire trained model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    /// Lambda — prior probability that two random records match.
    pub probability_two_random_records_match: f64,
    /// Bayes factor derived from lambda.
    pub prior_bayes_factor: f64,
    /// Log2 of the prior Bayes factor (the prior match weight).
    pub prior_match_weight: f64,
    /// Per-comparison summaries with trained level parameters.
    pub comparisons: Vec<ComparisonSummary>,
}

// ── Cell extraction helpers ──────────────────────────────────────────

fn cell_to_string(df: &DataFrame, col_name: &str, row: usize) -> Result<Option<String>> {
    let column = df
        .column(col_name)
        .map_err(|_| WeldrsError::Config(format!("Column '{col_name}' not found")))?;
    let val = column.get(row).map_err(|e| {
        WeldrsError::Config(format!("Row {row} out of bounds for '{col_name}': {e}"))
    })?;
    match val {
        AnyValue::Null => Ok(None),
        AnyValue::String(s) => Ok(Some(s.to_string())),
        AnyValue::StringOwned(s) => Ok(Some(s.to_string())),
        other => Ok(Some(format!("{other}"))),
    }
}

fn cell_to_i32(df: &DataFrame, col_name: &str, row: usize) -> Result<i32> {
    let column = df
        .column(col_name)
        .map_err(|_| WeldrsError::Config(format!("Column '{col_name}' not found")))?;
    let val = column.get(row).map_err(|e| {
        WeldrsError::Config(format!("Row {row} out of bounds for '{col_name}': {e}"))
    })?;
    match val {
        AnyValue::Int8(v) => Ok(v as i32),
        AnyValue::Int16(v) => Ok(v as i32),
        AnyValue::Int32(v) => Ok(v),
        AnyValue::Int64(v) => Ok(v as i32),
        AnyValue::UInt8(v) => Ok(v as i32),
        AnyValue::UInt16(v) => Ok(v as i32),
        AnyValue::UInt32(v) => Ok(v as i32),
        AnyValue::UInt64(v) => Ok(v as i32),
        other => Err(WeldrsError::Config(format!(
            "Expected integer in '{col_name}' row {row}, got {other:?}"
        ))),
    }
}

fn cell_to_f64(df: &DataFrame, col_name: &str, row: usize) -> Result<f64> {
    let column = df
        .column(col_name)
        .map_err(|_| WeldrsError::Config(format!("Column '{col_name}' not found")))?;
    let val = column.get(row).map_err(|e| {
        WeldrsError::Config(format!("Row {row} out of bounds for '{col_name}': {e}"))
    })?;
    match val {
        AnyValue::Float32(v) => Ok(v as f64),
        AnyValue::Float64(v) => Ok(v),
        AnyValue::Int8(v) => Ok(v as f64),
        AnyValue::Int16(v) => Ok(v as f64),
        AnyValue::Int32(v) => Ok(v as f64),
        AnyValue::Int64(v) => Ok(v as f64),
        AnyValue::UInt8(v) => Ok(v as f64),
        AnyValue::UInt16(v) => Ok(v as f64),
        AnyValue::UInt32(v) => Ok(v as f64),
        AnyValue::UInt64(v) => Ok(v as f64),
        other => Err(WeldrsError::Config(format!(
            "Expected numeric in '{col_name}' row {row}, got {other:?}"
        ))),
    }
}

// ── Public functions ─────────────────────────────────────────────────

/// Produce a waterfall breakdown for a single record pair.
///
/// # Errors
///
/// Returns an error if `row_index` is out of bounds or if required
/// columns (gamma, BF, unique ID, match_weight, match_probability)
/// are missing.
pub fn explain_pair(
    predictions: &DataFrame,
    row_index: usize,
    comparisons: &[Comparison],
    lambda: f64,
    gamma_prefix: &str,
    bf_prefix: &str,
    unique_id_column: &str,
) -> Result<WaterfallChart> {
    if row_index >= predictions.height() {
        return Err(WeldrsError::Config(format!(
            "Row index {row_index} out of bounds (DataFrame has {} rows)",
            predictions.height()
        )));
    }

    let uid_l_col = format!("{unique_id_column}_l");
    let uid_r_col = format!("{unique_id_column}_r");
    let unique_id_l =
        cell_to_string(predictions, &uid_l_col, row_index)?.unwrap_or_else(|| "null".to_string());
    let unique_id_r =
        cell_to_string(predictions, &uid_r_col, row_index)?.unwrap_or_else(|| "null".to_string());

    let mut steps = Vec::with_capacity(comparisons.len() + 1);

    // Prior step
    let prior_bf = probability::prob_to_bayes_factor(lambda);
    let prior_log2 = prior_bf.log2();
    let prior_prob = probability::bayes_factor_to_prob(2.0_f64.powf(prior_log2));
    steps.push(WaterfallStep {
        column_name: "Prior".to_string(),
        label: "Prior (lambda)".to_string(),
        comparison_vector_value: None,
        value_l: None,
        value_r: None,
        bayes_factor: prior_bf,
        log2_bayes_factor: prior_log2,
        m_probability: None,
        u_probability: None,
        cumulative_match_weight: prior_log2,
        cumulative_match_probability: prior_prob,
    });

    let mut cumulative_weight = prior_log2;

    // One step per comparison
    for comp in comparisons {
        let gamma_col = comp.gamma_column_name(gamma_prefix);
        let bf_col = comp.bf_column_name(bf_prefix);

        let gamma_val = cell_to_i32(predictions, &gamma_col, row_index)?;
        let bf_val = cell_to_f64(predictions, &bf_col, row_index)?;
        let log2_bf = bf_val.log2();

        // Find matching level
        let matched_level = comp
            .comparison_levels
            .iter()
            .find(|l| l.comparison_vector_value == gamma_val);

        let (m_prob, u_prob, label) = match matched_level {
            Some(level) if level.is_null_level => (None, None, level.label.clone()),
            Some(level) => (
                level.m_probability,
                level.u_probability,
                level.label.clone(),
            ),
            None => (None, None, format!("Unknown (gamma={gamma_val})")),
        };

        // Extract left/right field values from first input column
        let (value_l, value_r) = if let Some(input_col) = comp.input_columns.first() {
            let col_l = format!("{input_col}_l");
            let col_r = format!("{input_col}_r");
            let vl = cell_to_string(predictions, &col_l, row_index).unwrap_or(None);
            let vr = cell_to_string(predictions, &col_r, row_index).unwrap_or(None);
            (vl, vr)
        } else {
            (None, None)
        };

        cumulative_weight += log2_bf;
        let cumulative_prob = probability::bayes_factor_to_prob(2.0_f64.powf(cumulative_weight));

        steps.push(WaterfallStep {
            column_name: comp.output_column_name.clone(),
            label,
            comparison_vector_value: Some(gamma_val),
            value_l,
            value_r,
            bayes_factor: bf_val,
            log2_bayes_factor: log2_bf,
            m_probability: m_prob,
            u_probability: u_prob,
            cumulative_match_weight: cumulative_weight,
            cumulative_match_probability: cumulative_prob,
        });
    }

    let final_match_weight = cell_to_f64(predictions, "match_weight", row_index)?;
    let final_match_probability = cell_to_f64(predictions, "match_probability", row_index)?;

    Ok(WaterfallChart {
        unique_id_l,
        unique_id_r,
        steps,
        final_match_weight,
        final_match_probability,
    })
}

/// Produce waterfall breakdowns for multiple record pairs.
///
/// # Errors
///
/// Returns an error if any `row_index` is out of bounds or if required
/// columns are missing.
pub fn explain_pairs(
    predictions: &DataFrame,
    row_indices: &[usize],
    comparisons: &[Comparison],
    lambda: f64,
    gamma_prefix: &str,
    bf_prefix: &str,
    unique_id_column: &str,
) -> Result<Vec<WaterfallChart>> {
    row_indices
        .iter()
        .map(|&idx| {
            explain_pair(
                predictions,
                idx,
                comparisons,
                lambda,
                gamma_prefix,
                bf_prefix,
                unique_id_column,
            )
        })
        .collect()
}

/// Produce a structured summary of the trained model parameters.
pub fn model_summary(settings: &Settings) -> ModelSummary {
    let lambda = settings.probability_two_random_records_match;
    let prior_bf = probability::prob_to_bayes_factor(lambda);
    let prior_mw = prior_bf.log2();

    let comparisons = settings
        .comparisons
        .iter()
        .map(|comp| {
            let mut levels: Vec<LevelSummary> = comp
                .comparison_levels
                .iter()
                .map(|level| LevelSummary {
                    label: level.label.clone(),
                    comparison_vector_value: level.comparison_vector_value,
                    is_null_level: level.is_null_level,
                    m_probability: level.m_probability,
                    u_probability: level.u_probability,
                    bayes_factor: if level.is_null_level {
                        None
                    } else {
                        level.bayes_factor()
                    },
                    log2_bayes_factor: if level.is_null_level {
                        None
                    } else {
                        level.log2_bayes_factor()
                    },
                })
                .collect();

            // Sort: non-null levels by cv value descending (best match first), null last
            levels.sort_by(|a, b| match (a.is_null_level, b.is_null_level) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => b.comparison_vector_value.cmp(&a.comparison_vector_value),
            });

            ComparisonSummary {
                output_column_name: comp.output_column_name.clone(),
                description: comp.description.clone(),
                input_columns: comp.input_columns.clone(),
                levels,
            }
        })
        .collect();

    ModelSummary {
        probability_two_random_records_match: lambda,
        prior_bayes_factor: prior_bf,
        prior_match_weight: prior_mw,
        comparisons,
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocking::BlockingRule;
    use crate::comparison::ComparisonBuilder;
    use crate::predict;
    use crate::settings::LinkType;

    fn make_trained_settings() -> Settings {
        let mut comp_fn = ComparisonBuilder::new("first_name")
            .null_level()
            .exact_match_level()
            .else_level()
            .build();
        for level in &mut comp_fn.comparison_levels {
            if level.is_null_level {
                continue;
            }
            if level.comparison_vector_value == 1 {
                level.m_probability = Some(0.9);
                level.u_probability = Some(0.1);
            } else {
                level.m_probability = Some(0.1);
                level.u_probability = Some(0.9);
            }
        }

        let mut comp_sn = ComparisonBuilder::new("last_name")
            .null_level()
            .exact_match_level()
            .else_level()
            .build();
        for level in &mut comp_sn.comparison_levels {
            if level.is_null_level {
                continue;
            }
            if level.comparison_vector_value == 1 {
                level.m_probability = Some(0.85);
                level.u_probability = Some(0.05);
            } else {
                level.m_probability = Some(0.15);
                level.u_probability = Some(0.95);
            }
        }

        Settings::builder(LinkType::DedupeOnly)
            .comparison(comp_fn)
            .comparison(comp_sn)
            .probability_two_random_records_match(0.01)
            .blocking_rule(BlockingRule::on(&["last_name"]))
            .build()
            .unwrap()
    }

    fn make_predictions(settings: &Settings) -> DataFrame {
        let df = df!(
            "unique_id_l" => [1i64, 2, 3],
            "unique_id_r" => [4i64, 5, 6],
            "first_name_l" => ["Alice", "Bob", "Carol"],
            "first_name_r" => ["Alice", "Xavier", "Carol"],
            "last_name_l" => ["Smith", "Jones", "Smith"],
            "last_name_r" => ["Smith", "Jones", "Brown"],
            "gamma_first_name" => [1i8, 0, 1],
            "gamma_last_name" => [1i8, 1, 0],
        )
        .unwrap();

        predict::predict(
            df.lazy(),
            &settings.comparisons,
            settings.probability_two_random_records_match,
            &settings.gamma_prefix,
            &settings.bf_prefix,
            None,
            None,
        )
        .unwrap()
        .collect()
        .unwrap()
    }

    // ── model_summary tests ──────────────────────────────────────────

    #[test]
    fn test_model_summary_structure() {
        let settings = make_trained_settings();
        let summary = model_summary(&settings);
        assert_eq!(summary.comparisons.len(), 2);
        assert_eq!(summary.comparisons[0].output_column_name, "first_name");
        assert_eq!(summary.comparisons[1].output_column_name, "last_name");
        // Each comparison has 3 levels (null + exact + else)
        assert_eq!(summary.comparisons[0].levels.len(), 3);
        assert_eq!(summary.comparisons[1].levels.len(), 3);
    }

    #[test]
    fn test_model_summary_prior_values() {
        let settings = make_trained_settings();
        let summary = model_summary(&settings);

        let expected_bf = probability::prob_to_bayes_factor(0.01);
        let expected_mw = expected_bf.log2();

        assert!((summary.probability_two_random_records_match - 0.01).abs() < 1e-10);
        assert!((summary.prior_bayes_factor - expected_bf).abs() < 1e-10);
        assert!((summary.prior_match_weight - expected_mw).abs() < 1e-10);
    }

    #[test]
    fn test_model_summary_level_ordering() {
        let settings = make_trained_settings();
        let summary = model_summary(&settings);

        for comp in &summary.comparisons {
            // Last level should be null
            assert!(comp.levels.last().unwrap().is_null_level);
            // First level should have highest cv value among non-null
            let non_null: Vec<_> = comp.levels.iter().filter(|l| !l.is_null_level).collect();
            assert!(non_null[0].comparison_vector_value >= non_null[1].comparison_vector_value);
        }
    }

    #[test]
    fn test_model_summary_null_level_has_none() {
        let settings = make_trained_settings();
        let summary = model_summary(&settings);

        for comp in &summary.comparisons {
            let null_level = comp.levels.iter().find(|l| l.is_null_level).unwrap();
            assert!(null_level.m_probability.is_none());
            assert!(null_level.u_probability.is_none());
            assert!(null_level.bayes_factor.is_none());
            assert!(null_level.log2_bayes_factor.is_none());
        }
    }

    #[test]
    fn test_model_summary_serializes_to_json() {
        let settings = make_trained_settings();
        let summary = model_summary(&settings);
        let json = serde_json::to_string(&summary).unwrap();
        let restored: ModelSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.comparisons.len(), summary.comparisons.len());
        assert!(
            (restored.probability_two_random_records_match
                - summary.probability_two_random_records_match)
                .abs()
                < 1e-10
        );
    }

    // ── waterfall tests ──────────────────────────────────────────────

    #[test]
    fn test_waterfall_step_count() {
        let settings = make_trained_settings();
        let predictions = make_predictions(&settings);
        let chart = explain_pair(
            &predictions,
            0,
            &settings.comparisons,
            settings.probability_two_random_records_match,
            &settings.gamma_prefix,
            &settings.bf_prefix,
            &settings.unique_id_column,
        )
        .unwrap();
        // Prior + 2 comparisons = 3 steps
        assert_eq!(chart.steps.len(), 3);
    }

    #[test]
    fn test_waterfall_prior_step() {
        let settings = make_trained_settings();
        let predictions = make_predictions(&settings);
        let chart = explain_pair(
            &predictions,
            0,
            &settings.comparisons,
            settings.probability_two_random_records_match,
            &settings.gamma_prefix,
            &settings.bf_prefix,
            &settings.unique_id_column,
        )
        .unwrap();

        let prior = &chart.steps[0];
        assert_eq!(prior.column_name, "Prior");
        assert_eq!(prior.label, "Prior (lambda)");
        assert!(prior.comparison_vector_value.is_none());
        assert!(prior.value_l.is_none());
        assert!(prior.value_r.is_none());

        let expected_bf = probability::prob_to_bayes_factor(0.01);
        assert!((prior.bayes_factor - expected_bf).abs() < 1e-10);
        assert!((prior.log2_bayes_factor - expected_bf.log2()).abs() < 1e-10);
    }

    #[test]
    fn test_waterfall_cumulative_progression() {
        let settings = make_trained_settings();
        let predictions = make_predictions(&settings);
        let chart = explain_pair(
            &predictions,
            0,
            &settings.comparisons,
            settings.probability_two_random_records_match,
            &settings.gamma_prefix,
            &settings.bf_prefix,
            &settings.unique_id_column,
        )
        .unwrap();

        let mut running = 0.0;
        for step in &chart.steps {
            running += step.log2_bayes_factor;
            assert!(
                (step.cumulative_match_weight - running).abs() < 1e-10,
                "Cumulative should equal running sum at step '{}'",
                step.column_name
            );
        }
    }

    #[test]
    fn test_waterfall_final_values_match() {
        let settings = make_trained_settings();
        let predictions = make_predictions(&settings);
        let chart = explain_pair(
            &predictions,
            0,
            &settings.comparisons,
            settings.probability_two_random_records_match,
            &settings.gamma_prefix,
            &settings.bf_prefix,
            &settings.unique_id_column,
        )
        .unwrap();

        let df_weight = cell_to_f64(&predictions, "match_weight", 0).unwrap();
        let df_prob = cell_to_f64(&predictions, "match_probability", 0).unwrap();

        assert!(
            (chart.final_match_weight - df_weight).abs() < 1e-6,
            "final_match_weight {} != DataFrame {}",
            chart.final_match_weight,
            df_weight
        );
        assert!(
            (chart.final_match_probability - df_prob).abs() < 1e-6,
            "final_match_probability {} != DataFrame {}",
            chart.final_match_probability,
            df_prob
        );
    }

    #[test]
    fn test_waterfall_exact_match_positive() {
        let settings = make_trained_settings();
        let predictions = make_predictions(&settings);
        // Row 0: both first_name and last_name are exact matches (gamma=1)
        let chart = explain_pair(
            &predictions,
            0,
            &settings.comparisons,
            settings.probability_two_random_records_match,
            &settings.gamma_prefix,
            &settings.bf_prefix,
            &settings.unique_id_column,
        )
        .unwrap();

        // Both comparison steps should have positive log2 BF
        for step in &chart.steps[1..] {
            assert!(
                step.log2_bayes_factor > 0.0,
                "Exact match step '{}' should have positive log2_bf, got {}",
                step.column_name,
                step.log2_bayes_factor
            );
        }
    }

    #[test]
    fn test_waterfall_non_match_negative() {
        let settings = make_trained_settings();
        let predictions = make_predictions(&settings);
        // Row 2: last_name is non-match (gamma=0, Smith vs Brown)
        let chart = explain_pair(
            &predictions,
            2,
            &settings.comparisons,
            settings.probability_two_random_records_match,
            &settings.gamma_prefix,
            &settings.bf_prefix,
            &settings.unique_id_column,
        )
        .unwrap();

        // last_name step (index 2) should have negative log2 BF
        let last_name_step = &chart.steps[2];
        assert_eq!(last_name_step.column_name, "last_name");
        assert!(
            last_name_step.log2_bayes_factor < 0.0,
            "Non-match step should have negative log2_bf, got {}",
            last_name_step.log2_bayes_factor
        );
    }

    #[test]
    fn test_waterfall_null_level_neutral() {
        // Build a comparison with null gamma => BF=1.0
        let mut comp = ComparisonBuilder::new("col")
            .null_level()
            .exact_match_level()
            .else_level()
            .build();
        for level in &mut comp.comparison_levels {
            if !level.is_null_level {
                if level.comparison_vector_value == 1 {
                    level.m_probability = Some(0.9);
                    level.u_probability = Some(0.1);
                } else {
                    level.m_probability = Some(0.1);
                    level.u_probability = Some(0.9);
                }
            }
        }

        let df = df!(
            "unique_id_l" => [1i64],
            "unique_id_r" => [2i64],
            "col_l" => [None::<&str>],
            "col_r" => [None::<&str>],
            "gamma_col" => [-1i8],
        )
        .unwrap();

        let scored = predict::predict(
            df.lazy(),
            &[comp.clone()],
            0.01,
            "gamma_",
            "bf_",
            None,
            None,
        )
        .unwrap()
        .collect()
        .unwrap();

        let chart = explain_pair(&scored, 0, &[comp], 0.01, "gamma_", "bf_", "unique_id").unwrap();

        let col_step = &chart.steps[1];
        assert!((col_step.bayes_factor - 1.0).abs() < 1e-10);
        assert!(col_step.log2_bayes_factor.abs() < 1e-10);
    }

    #[test]
    fn test_waterfall_value_extraction() {
        let settings = make_trained_settings();
        let predictions = make_predictions(&settings);
        let chart = explain_pair(
            &predictions,
            0,
            &settings.comparisons,
            settings.probability_two_random_records_match,
            &settings.gamma_prefix,
            &settings.bf_prefix,
            &settings.unique_id_column,
        )
        .unwrap();

        // Row 0: first_name Alice/Alice, last_name Smith/Smith
        let fn_step = &chart.steps[1];
        assert_eq!(fn_step.value_l.as_deref(), Some("Alice"));
        assert_eq!(fn_step.value_r.as_deref(), Some("Alice"));

        let sn_step = &chart.steps[2];
        assert_eq!(sn_step.value_l.as_deref(), Some("Smith"));
        assert_eq!(sn_step.value_r.as_deref(), Some("Smith"));
    }

    #[test]
    fn test_explain_pairs_batch() {
        let settings = make_trained_settings();
        let predictions = make_predictions(&settings);
        let charts = explain_pairs(
            &predictions,
            &[0, 1, 2],
            &settings.comparisons,
            settings.probability_two_random_records_match,
            &settings.gamma_prefix,
            &settings.bf_prefix,
            &settings.unique_id_column,
        )
        .unwrap();
        assert_eq!(charts.len(), 3);
    }

    #[test]
    fn test_explain_pair_out_of_bounds() {
        let settings = make_trained_settings();
        let predictions = make_predictions(&settings);
        let result = explain_pair(
            &predictions,
            999,
            &settings.comparisons,
            settings.probability_two_random_records_match,
            &settings.gamma_prefix,
            &settings.bf_prefix,
            &settings.unique_id_column,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_waterfall_unique_ids() {
        let settings = make_trained_settings();
        let predictions = make_predictions(&settings);
        let chart = explain_pair(
            &predictions,
            0,
            &settings.comparisons,
            settings.probability_two_random_records_match,
            &settings.gamma_prefix,
            &settings.bf_prefix,
            &settings.unique_id_column,
        )
        .unwrap();
        assert_eq!(chart.unique_id_l, "1");
        assert_eq!(chart.unique_id_r, "4");
    }
}
