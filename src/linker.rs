//! High-level pipeline orchestrator.
//!
//! The [`Linker`] struct is the main entry point for most users. It holds
//! the model [`Settings`] and exposes methods for training, prediction,
//! clustering, explanation, and serialization.

use polars::prelude::*;

use crate::blocking::{self, BlockingRule};
use crate::clustering;
use crate::comparison_vectors;
use crate::em;
use crate::error::Result;
use crate::estimate_lambda;
use crate::estimate_u;
use crate::explain;
use crate::predict;
use crate::settings::Settings;

/// High-level orchestrator for the weldrs record-linkage pipeline.
///
/// Holds the model settings (including trained parameters) and provides
/// methods for training, prediction, and clustering.
pub struct Linker {
    /// The model configuration and trained parameters.
    pub settings: Settings,
}

impl Linker {
    /// Create a new linker with the given settings.
    pub fn new(settings: Settings) -> Result<Self> {
        Ok(Self { settings })
    }

    // ── Training ──────────────────────────────────────────────────────

    /// Estimate lambda (probability two random records match) using
    /// deterministic rules.
    pub fn estimate_probability_two_random_records_match(
        &mut self,
        df: &LazyFrame,
        deterministic_rules: &[BlockingRule],
        recall: f64,
    ) -> Result<f64> {
        let lambda = estimate_lambda::estimate_probability_two_random_records_match(
            df,
            deterministic_rules,
            &self.settings.link_type,
            &self.settings.unique_id_column,
            recall,
        )?;
        self.settings.probability_two_random_records_match = lambda;
        Ok(lambda)
    }

    /// Estimate u-probabilities from random record pairs.
    pub fn estimate_u_using_random_sampling(
        &mut self,
        df: &LazyFrame,
        max_pairs: usize,
    ) -> Result<()> {
        estimate_u::estimate_u_using_random_sampling(
            df,
            &mut self.settings.comparisons,
            max_pairs,
            &self.settings.gamma_prefix,
            &self.settings.unique_id_column,
        )
    }

    /// Train m/u parameters using the EM algorithm, blocking on the given rule.
    ///
    /// Comparisons whose input columns overlap with the blocking rule columns
    /// are excluded from EM estimation (they always agree under that block).
    pub fn estimate_parameters_using_em(
        &mut self,
        df: &LazyFrame,
        blocking_rule: &BlockingRule,
    ) -> Result<()> {
        // Generate blocked pairs using the training blocking rule.
        let blocked = blocking::generate_blocked_pairs(
            df,
            std::slice::from_ref(blocking_rule),
            &self.settings.link_type,
            &self.settings.unique_id_column,
        )?;

        // Compute comparison vectors.
        let cv = comparison_vectors::compute_comparison_vectors(
            blocked,
            &self.settings.comparisons,
            &self.settings.gamma_prefix,
        )?;

        // Run EM. Columns that overlap with the blocking rule are fixed.
        let columns_to_fix = blocking_rule.columns.clone();

        let results = em::expectation_maximization(
            &cv,
            self.settings.comparisons.clone(),
            self.settings.probability_two_random_records_match,
            &self.settings.training,
            &self.settings.gamma_prefix,
            &columns_to_fix,
        )?;

        // Apply the final EM result back into settings.
        if let Some(last) = results.last() {
            // Update only the comparisons whose parameters were estimated
            // (not fixed).
            for (i, updated_comp) in last.comparisons.iter().enumerate() {
                let orig = &mut self.settings.comparisons[i];
                for (j, updated_level) in updated_comp.comparison_levels.iter().enumerate() {
                    let orig_level = &mut orig.comparison_levels[j];

                    if !updated_level.fix_m_probability {
                        orig_level.m_probability = updated_level.m_probability;
                    }
                    if !updated_level.fix_u_probability {
                        orig_level.u_probability = updated_level.u_probability;
                    }
                }
            }
        }

        Ok(())
    }

    // ── Inference ─────────────────────────────────────────────────────

    /// Score record pairs using the trained model.
    ///
    /// Generates blocked pairs, computes comparison vectors, and applies the
    /// Fellegi-Sunter scoring model. Returns a `LazyFrame` with
    /// `match_weight` and `match_probability` columns.
    pub fn predict(
        &self,
        df: &LazyFrame,
        threshold_match_weight: Option<f64>,
    ) -> Result<LazyFrame> {
        let blocked = blocking::generate_blocked_pairs(
            df,
            &self.settings.blocking_rules,
            &self.settings.link_type,
            &self.settings.unique_id_column,
        )?;

        let cv = comparison_vectors::compute_comparison_vectors(
            blocked,
            &self.settings.comparisons,
            &self.settings.gamma_prefix,
        )?;

        predict::predict(
            cv,
            &self.settings.comparisons,
            self.settings.probability_two_random_records_match,
            &self.settings.gamma_prefix,
            &self.settings.bf_prefix,
            None,
            threshold_match_weight,
        )
    }

    // ── Clustering ────────────────────────────────────────────────────

    /// Cluster pairwise predictions into groups of linked records.
    pub fn cluster_pairwise_predictions(
        &self,
        predictions: &DataFrame,
        threshold: f64,
    ) -> Result<DataFrame> {
        let uid_l = format!("{}_l", self.settings.unique_id_column);
        let uid_r = format!("{}_r", self.settings.unique_id_column);
        clustering::cluster_pairwise_predictions(predictions, threshold, &uid_l, &uid_r)
    }

    // ── Explanation ────────────────────────────────────────────────────

    /// Explain why a single record pair received its score.
    pub fn explain_pair(
        &self,
        predictions: &DataFrame,
        row_index: usize,
    ) -> Result<explain::WaterfallChart> {
        explain::explain_pair(
            predictions,
            row_index,
            &self.settings.comparisons,
            self.settings.probability_two_random_records_match,
            &self.settings.gamma_prefix,
            &self.settings.bf_prefix,
            &self.settings.unique_id_column,
        )
    }

    /// Explain why multiple record pairs received their scores.
    pub fn explain_pairs(
        &self,
        predictions: &DataFrame,
        row_indices: &[usize],
    ) -> Result<Vec<explain::WaterfallChart>> {
        explain::explain_pairs(
            predictions,
            row_indices,
            &self.settings.comparisons,
            self.settings.probability_two_random_records_match,
            &self.settings.gamma_prefix,
            &self.settings.bf_prefix,
            &self.settings.unique_id_column,
        )
    }

    /// Produce a structured summary of the trained model parameters.
    pub fn model_summary(&self) -> explain::ModelSummary {
        explain::model_summary(&self.settings)
    }

    // ── Serialization ─────────────────────────────────────────────────

    /// Serialize the current settings (including trained parameters) to JSON.
    pub fn save_settings_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.settings).map_err(Into::into)
    }

    /// Load a linker from previously saved JSON settings.
    pub fn load_settings_json(json: &str) -> Result<Self> {
        let settings: Settings = serde_json::from_str(json)?;
        Self::new(settings)
    }
}

// ── Visualization (feature-gated) ─────────────────────────────────

#[cfg(feature = "visualize")]
impl Linker {
    /// Render a waterfall chart for a single record pair as an SVG string.
    pub fn waterfall_chart_svg(
        &self,
        predictions: &DataFrame,
        row_index: usize,
        options: &crate::visualize::ChartOptions,
    ) -> Result<String> {
        let waterfall = self.explain_pair(predictions, row_index)?;
        crate::visualize::waterfall_chart_svg(&waterfall, options)
    }

    /// Render a match weights chart for the trained model as an SVG string.
    pub fn match_weights_chart_svg(
        &self,
        options: &crate::visualize::ChartOptions,
    ) -> Result<String> {
        let summary = self.model_summary();
        crate::visualize::match_weights_chart_svg(&summary, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::LinkType;
    use crate::test_helpers;

    fn make_linker() -> Linker {
        let settings = crate::settings::Settings::builder(LinkType::DedupeOnly)
            .comparison(test_helpers::exact_match_comparison("first_name"))
            .comparison(test_helpers::exact_match_comparison("surname"))
            .blocking_rule(BlockingRule::on(&["surname"]))
            .build()
            .unwrap();
        Linker::new(settings).unwrap()
    }

    #[test]
    fn test_linker_new() {
        let linker = make_linker();
        assert_eq!(linker.settings.comparisons.len(), 2);
        assert_eq!(linker.settings.link_type, LinkType::DedupeOnly);
    }

    #[test]
    fn test_save_load_settings_json() {
        let linker = make_linker();
        let json = linker.save_settings_json().unwrap();
        let restored = Linker::load_settings_json(&json).unwrap();

        assert_eq!(
            restored.settings.comparisons.len(),
            linker.settings.comparisons.len()
        );
        assert_eq!(restored.settings.link_type, linker.settings.link_type);
        assert_eq!(
            restored.settings.unique_id_column,
            linker.settings.unique_id_column
        );
    }

    #[test]
    fn test_estimate_lambda_updates_settings() {
        let mut linker = make_linker();
        let df = test_helpers::make_test_df().lazy();

        let initial_lambda = linker.settings.probability_two_random_records_match;

        linker
            .estimate_probability_two_random_records_match(
                &df,
                &[BlockingRule::on(&["first_name", "surname"])],
                1.0,
            )
            .unwrap();

        assert_ne!(
            linker.settings.probability_two_random_records_match, initial_lambda,
            "Lambda should change after estimation"
        );
    }

    #[test]
    fn test_predict_returns_scored_pairs() {
        let mut linker = make_linker();
        let df = test_helpers::make_test_df().lazy();

        // Estimate u values so BFs are reasonable
        linker.estimate_u_using_random_sampling(&df, 100).unwrap();

        let predictions = linker.predict(&df, None).unwrap().collect().unwrap();

        let col_names: Vec<&str> = predictions
            .get_column_names()
            .into_iter()
            .map(|s| s.as_str())
            .collect();
        assert!(col_names.contains(&"match_weight"));
        assert!(col_names.contains(&"match_probability"));
        assert!(predictions.height() > 0);
    }
}
