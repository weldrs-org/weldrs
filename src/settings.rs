//! Model settings, training parameters, and builder.
//!
//! [`Settings`] holds the complete configuration for a weldrs model: link type,
//! comparisons, blocking rules, trained parameters, and column naming
//! conventions. Use [`Settings::builder`] to construct one.
//!
//! Settings are serializable via serde, so a trained model can be saved to
//! JSON and loaded later via
//! [`Linker::save_settings_json`](crate::linker::Linker::save_settings_json) /
//! [`Linker::load_settings_json`](crate::linker::Linker::load_settings_json).
//!
//! See [`comparison`](crate::comparison) for building comparisons and
//! [`blocking`](crate::blocking) for blocking rules.
//!
//! # Example
//!
//! ```
//! use weldrs::comparison::ComparisonBuilder;
//! use weldrs::blocking::BlockingRule;
//! use weldrs::settings::{LinkType, Settings};
//!
//! let settings = Settings::builder(LinkType::DedupeOnly)
//!     .comparison(
//!         ComparisonBuilder::new("first_name")
//!             .null_level()
//!             .exact_match_level()
//!             .else_level()
//!             .build()
//!             .unwrap(),
//!     )
//!     .blocking_rule(BlockingRule::on(&["last_name"]))
//!     .build()
//!     .unwrap();
//!
//! assert_eq!(settings.comparisons.len(), 1);
//! assert_eq!(settings.blocking_rules.len(), 1);
//! ```

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
    /// Whether to store a snapshot of all comparisons at each iteration.
    /// When `false`, only the final result is kept, avoiding a `clone()` per
    /// iteration. Default: `true` for backward compatibility.
    #[serde(default = "default_store_history")]
    pub store_history: bool,
}

fn default_store_history() -> bool {
    true
}

impl Default for TrainingSettings {
    fn default() -> Self {
        Self {
            em_convergence: 0.0001,
            max_iterations: 25,
            store_history: true,
        }
    }
}

fn default_version() -> u32 {
    1
}

/// Full settings for a weldrs model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Schema version for forward-compatible deserialization.
    /// Old JSON without this field will default to version 1.
    #[serde(default = "default_version")]
    pub version: u32,
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn version_defaults_to_one_when_missing() {
        // JSON payload intentionally omits the `version` field to simulate
        // settings saved by older versions of the library.
        let json = r#"
        {
            "link_type": "DedupeOnly",
            "comparisons": [],
            "blocking_rules": [],
            "probability_two_random_records_match": 0.001,
            "unique_id_column": "id",
            "source_dataset_column": null,
            "training": {
                "em_convergence": 0.0001,
                "max_iterations": 10,
                "store_history": true
            },
            "gamma_prefix": "gamma_",
            "bf_prefix": "bf_"
        }
        "#;

        let settings: Settings = serde_json::from_str(json).expect("deserialization should succeed");
        assert_eq!(settings.version, 1);
    }
}

impl Settings {
    /// Start building a [`Settings`] value.
    ///
    /// # Examples
    ///
    /// ```
    /// use weldrs::comparison::ComparisonBuilder;
    /// use weldrs::blocking::BlockingRule;
    /// use weldrs::settings::{LinkType, Settings};
    ///
    /// let settings = Settings::builder(LinkType::DedupeOnly)
    ///     .comparison(
    ///         ComparisonBuilder::new("name")
    ///             .null_level()
    ///             .exact_match_level()
    ///             .else_level()
    ///             .build()
    ///             .unwrap(),
    ///     )
    ///     .blocking_rule(BlockingRule::on(&["name"]))
    ///     .probability_two_random_records_match(0.01)
    ///     .build()
    ///     .unwrap();
    /// ```
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
    ///
    /// # Errors
    ///
    /// Returns [`WeldrsError::Config`] if
    /// no comparisons have been added.
    pub fn build(self) -> Result<Settings> {
        if self.comparisons.is_empty() {
            return Err(WeldrsError::Config(
                "At least one comparison is required".into(),
            ));
        }

        // Validate that input column names don't collide with generated
        // suffixed/prefixed names.
        {
            let mut suffixed: std::collections::HashSet<String> = std::collections::HashSet::new();
            for comp in &self.comparisons {
                for col in &comp.input_columns {
                    suffixed.insert(format!("{col}_l"));
                    suffixed.insert(format!("{col}_r"));
                }
            }
            for comp in &self.comparisons {
                for col in &comp.input_columns {
                    if suffixed.contains(col.as_str()) {
                        return Err(WeldrsError::Config(format!(
                            "Input column '{col}' collides with a generated suffixed column name. \
                             Avoid columns ending in '_l' or '_r'."
                        )));
                    }
                }
                let gamma_name = format!("{}{}", self.gamma_prefix, comp.output_column_name);
                if suffixed.contains(&gamma_name) {
                    return Err(WeldrsError::Config(format!(
                        "Generated gamma column '{gamma_name}' collides with a suffixed input column."
                    )));
                }
            }
        }

        Ok(Settings {
            version: default_version(),
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
    use crate::comparison::ComparisonBuilder;
    use crate::error::WeldrsError;

    fn simple_comparison(column: &str) -> Comparison {
        ComparisonBuilder::new(column)
            .null_level()
            .exact_match_level()
            .else_level()
            .build()
            .expect("comparison should build successfully")
    }

    #[test]
    fn build_rejects_input_suffix_collisions() {
        // One comparison uses "name", another uses "name_l".
        // The suffixed set will contain "name_l" from the first comparison,
        // so the second comparison's input column "name_l" collides.
        let settings_result = Settings::builder(LinkType::DedupeOnly)
            .comparison(simple_comparison("name"))
            .comparison(simple_comparison("name_l"))
            .build();

        match settings_result {
            Err(WeldrsError::Config(_)) => {}
            other => panic!("expected WeldrsError::Config due to input suffix collision, got: {:?}", other),
        }
    }

    #[test]
    fn build_rejects_gamma_column_collisions_with_suffixed_inputs() {
        // Construct comparisons such that a generated gamma column name
        // collides with a suffixed input column name.
        //
        // Assuming the default gamma_prefix "gamma_" and that the output
        // column name is derived from the input column:
        // - For a comparison on "gamma_col", a suffixed name "gamma_col_l"
        //   will be generated.
        // - For a comparison on "col_l", the gamma column will be
        //   "gamma_col_l", colliding with the suffixed name above.
        let settings_result = Settings::builder(LinkType::DedupeOnly)
            .comparison(simple_comparison("gamma_col"))
            .comparison(simple_comparison("col_l"))
            .build();

        match settings_result {
            Err(WeldrsError::Config(_)) => {}
            other => panic!(
                "expected WeldrsError::Config due to gamma/suffixed column collision, got: {:?}",
                other
            ),
        }
    }
}
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
                ..Default::default()
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
            .blocking_rule(BlockingRule::on(&["last_name"]))
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
