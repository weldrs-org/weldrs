//! Model settings, training parameters, and builder.
//!
//! [`Settings`] holds the complete configuration for a weldrs model: link type,
//! comparisons, blocking rules, trained parameters, and column naming
//! conventions. Use [`Settings::builder`] to construct one.

use serde::{Deserialize, Serialize};

use crate::blocking::BlockingRule;
use crate::comparison::Comparison;
use crate::error::{Result, WeldrsError};

/// The type of record linkage being performed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LinkType {
    /// Deduplication within a single dataset.
    DedupeOnly,
    /// Linking between two distinct datasets.
    LinkOnly,
    /// Linking and deduplication combined.
    LinkAndDedupe,
}

/// Parameters controlling the EM training loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSettings {
    /// Maximum absolute change in any parameter before convergence is declared.
    pub em_convergence: f64,
    /// Maximum number of EM iterations.
    pub max_iterations: usize,
}

impl Default for TrainingSettings {
    fn default() -> Self {
        Self {
            em_convergence: 0.0001,
            max_iterations: 25,
        }
    }
}

/// Full settings for a weldrs model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Whether this is deduplication, linkage, or both.
    pub link_type: LinkType,
    /// The comparisons that define how record pairs are scored.
    pub comparisons: Vec<Comparison>,
    /// Blocking rules used during prediction to generate candidate pairs.
    pub blocking_rules: Vec<BlockingRule>,
    /// Prior probability that two randomly chosen records are a match (lambda).
    pub probability_two_random_records_match: f64,
    /// Name of the column containing unique record identifiers.
    pub unique_id_column: String,
    /// Optional column identifying which source dataset a record came from
    /// (used in `LinkOnly` mode).
    pub source_dataset_column: Option<String>,
    /// Parameters controlling the EM training loop.
    pub training: TrainingSettings,
    /// Prefix for gamma (comparison vector) column names (default `"gamma_"`).
    pub gamma_prefix: String,
    /// Prefix for Bayes factor column names (default `"bf_"`).
    pub bf_prefix: String,
}

impl Settings {
    /// Start building a [`Settings`] value.
    pub fn builder(link_type: LinkType) -> SettingsBuilder {
        SettingsBuilder::new(link_type)
    }
}

/// Ergonomic builder for [`Settings`].
pub struct SettingsBuilder {
    link_type: LinkType,
    comparisons: Vec<Comparison>,
    blocking_rules: Vec<BlockingRule>,
    probability_two_random_records_match: f64,
    unique_id_column: String,
    source_dataset_column: Option<String>,
    training: TrainingSettings,
    gamma_prefix: String,
    bf_prefix: String,
}

impl SettingsBuilder {
    fn new(link_type: LinkType) -> Self {
        Self {
            link_type,
            comparisons: Vec::new(),
            blocking_rules: Vec::new(),
            probability_two_random_records_match: 0.0001,
            unique_id_column: "unique_id".to_string(),
            source_dataset_column: None,
            training: TrainingSettings::default(),
            gamma_prefix: "gamma_".to_string(),
            bf_prefix: "bf_".to_string(),
        }
    }

    /// Add a comparison to the model.
    pub fn comparison(mut self, comparison: Comparison) -> Self {
        self.comparisons.push(comparison);
        self
    }

    /// Add a blocking rule used during prediction to generate candidate pairs.
    pub fn blocking_rule(mut self, rule: BlockingRule) -> Self {
        self.blocking_rules.push(rule);
        self
    }

    /// Set the prior probability that two random records match (lambda).
    /// Default: `0.0001`.
    pub fn probability_two_random_records_match(mut self, prob: f64) -> Self {
        self.probability_two_random_records_match = prob;
        self
    }

    /// Set the unique identifier column name. Default: `"unique_id"`.
    pub fn unique_id_column(mut self, col: &str) -> Self {
        self.unique_id_column = col.to_string();
        self
    }

    /// Set the source dataset column name (for `LinkOnly` mode).
    pub fn source_dataset_column(mut self, col: &str) -> Self {
        self.source_dataset_column = Some(col.to_string());
        self
    }

    /// Override the default EM training parameters.
    pub fn training_settings(mut self, training: TrainingSettings) -> Self {
        self.training = training;
        self
    }

    /// Set the prefix for gamma column names. Default: `"gamma_"`.
    pub fn gamma_prefix(mut self, prefix: &str) -> Self {
        self.gamma_prefix = prefix.to_string();
        self
    }

    /// Set the prefix for Bayes factor column names. Default: `"bf_"`.
    pub fn bf_prefix(mut self, prefix: &str) -> Self {
        self.bf_prefix = prefix.to_string();
        self
    }

    /// Build the [`Settings`]. Returns an error if no comparisons were added.
    pub fn build(self) -> Result<Settings> {
        if self.comparisons.is_empty() {
            return Err(WeldrsError::Config(
                "At least one comparison is required".into(),
            ));
        }
        Ok(Settings {
            link_type: self.link_type,
            comparisons: self.comparisons,
            blocking_rules: self.blocking_rules,
            probability_two_random_records_match: self.probability_two_random_records_match,
            unique_id_column: self.unique_id_column,
            source_dataset_column: self.source_dataset_column,
            training: self.training,
            gamma_prefix: self.gamma_prefix,
            bf_prefix: self.bf_prefix,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers;

    #[test]
    fn test_builder_defaults() {
        let comp = test_helpers::exact_match_comparison("name");
        let settings = Settings::builder(LinkType::DedupeOnly)
            .comparison(comp)
            .build()
            .unwrap();

        assert_eq!(settings.unique_id_column, "unique_id");
        assert_eq!(settings.gamma_prefix, "gamma_");
        assert_eq!(settings.bf_prefix, "bf_");
        assert!((settings.probability_two_random_records_match - 0.0001).abs() < 1e-10);
        assert!((settings.training.em_convergence - 0.0001).abs() < 1e-10);
        assert_eq!(settings.training.max_iterations, 25);
    }

    #[test]
    fn test_builder_custom_values() {
        let comp = test_helpers::exact_match_comparison("name");
        let settings = Settings::builder(LinkType::LinkOnly)
            .comparison(comp)
            .unique_id_column("record_id")
            .source_dataset_column("source")
            .probability_two_random_records_match(0.01)
            .gamma_prefix("g_")
            .bf_prefix("bayes_")
            .training_settings(TrainingSettings {
                em_convergence: 0.001,
                max_iterations: 50,
            })
            .blocking_rule(BlockingRule::on(&["city"]))
            .build()
            .unwrap();

        assert_eq!(settings.link_type, LinkType::LinkOnly);
        assert_eq!(settings.unique_id_column, "record_id");
        assert_eq!(settings.source_dataset_column.as_deref(), Some("source"));
        assert!((settings.probability_two_random_records_match - 0.01).abs() < 1e-10);
        assert_eq!(settings.gamma_prefix, "g_");
        assert_eq!(settings.bf_prefix, "bayes_");
        assert!((settings.training.em_convergence - 0.001).abs() < 1e-10);
        assert_eq!(settings.training.max_iterations, 50);
        assert_eq!(settings.blocking_rules.len(), 1);
    }

    #[test]
    fn test_builder_empty_comparisons_errors() {
        let result = Settings::builder(LinkType::DedupeOnly).build();
        assert!(result.is_err());
        match result.unwrap_err() {
            WeldrsError::Config(_) => {}
            other => panic!("Expected Config error, got: {other:?}"),
        }
    }

    #[test]
    fn test_settings_serde_roundtrip() {
        let comp = test_helpers::fuzzy_comparison("name", 0.85);
        let settings = Settings::builder(LinkType::DedupeOnly)
            .comparison(comp)
            .blocking_rule(BlockingRule::on(&["surname"]))
            .build()
            .unwrap();

        let json = serde_json::to_string(&settings).unwrap();
        let restored: Settings = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.link_type, settings.link_type);
        assert_eq!(restored.comparisons.len(), settings.comparisons.len());
        assert_eq!(restored.blocking_rules.len(), settings.blocking_rules.len());
        assert_eq!(restored.unique_id_column, settings.unique_id_column);
    }
}
