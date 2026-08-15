//! High-level pipeline orchestrator.
//!
//! The [`Linker`] struct is the main entry point for most users. It holds
//! the model [`Settings`] and exposes methods for training, prediction,
//! clustering, explanation, and serialization.

use std::collections::HashSet;

use polars::prelude::*;

use crate::blocking::{self, BlockingRule};
use crate::blocking_analysis;
use crate::clustering;
use crate::comparison_vectors;
use crate::em;
use crate::error::{Result, WeldrsError};
use crate::estimate_lambda;
use crate::estimate_u;
use crate::evaluation;
use crate::explain;
use crate::predict;
use crate::settings::Settings;

/// High-level orchestrator for the weldrs record-linkage pipeline.
///
/// Holds the model settings (including trained parameters) and provides
/// methods for training, prediction, and clustering.
pub struct Linker {
    /// Model settings. Access via [`Linker::settings`] / [`Linker::settings_mut`].
    settings: Settings,
}

impl Linker {
    /// Read-only access to the model settings.
    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Mutable access to the model settings for advanced users who need
    /// to modify parameters directly.
    pub fn settings_mut(&mut self) -> &mut Settings {
        &mut self.settings
    }
}

fn lazy_row_count(lf: &LazyFrame) -> Result<usize> {
    let row_count = lf
        .clone()
        .select([len().alias("__n_rows")])
        .collect()
        .map_err(|e| WeldrsError::Training {
            stage: "linker",
            message: format!("Failed to count candidate pairs: {e}"),
        })?;
    let series = row_count
        .column("__n_rows")
        .map_err(|e| WeldrsError::Training {
            stage: "linker",
            message: format!("Missing row-count column: {e}"),
        })?;
    let cast = series
        .cast(&DataType::UInt64)
        .map_err(|e| WeldrsError::Training {
            stage: "linker",
            message: format!("Row-count cast failed: {e}"),
        })?;
    let n = cast
        .u64()
        .map_err(|e| WeldrsError::Training {
            stage: "linker",
            message: format!("Row-count type error: {e}"),
        })?
        .get(0)
        .ok_or_else(|| WeldrsError::Training {
            stage: "linker",
            message: "Row-count query returned no rows".into(),
        })?;
    usize::try_from(n).map_err(|_| WeldrsError::Training {
        stage: "linker",
        message: "Row count exceeds usize".into(),
    })
}

impl Linker {
    /// Create a new linker with the given settings.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for forward compatibility.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use weldrs::comparison::ComparisonBuilder;
    /// use weldrs::prelude::*;
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
    ///     .build()
    ///     .unwrap();
    ///
    /// let linker = Linker::new(settings).unwrap();
    /// ```
    pub fn new(settings: Settings) -> Result<Self> {
        Ok(Self { settings })
    }

    // ── Training ──────────────────────────────────────────────────────

    /// Estimate lambda (probability two random records match) using
    /// deterministic rules.
    ///
    /// The deterministic rules should be high-confidence blocking rules
    /// (e.g., exact match on multiple columns) that identify "certain"
    /// matches. The `recall` parameter (0.0–1.0) adjusts for the
    /// assumed recall of these rules.
    ///
    /// # Errors
    ///
    /// Returns [`WeldrsError::Config`] if
    /// `deterministic_rules` is empty. Returns
    /// [`WeldrsError::Training`] if the
    /// DataFrame has fewer than 2 records.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use weldrs::prelude::*;
    /// # let mut linker: Linker = todo!();
    /// # use polars::prelude::IntoLazy;
    /// # let lf = polars::prelude::DataFrame::empty().lazy();
    /// linker.estimate_probability_two_random_records_match(
    ///     &lf,
    ///     &[BlockingRule::on(&["first_name", "last_name"])],
    ///     1.0, // assume 100% recall
    /// ).unwrap();
    /// ```
    pub fn estimate_probability_two_random_records_match(
        &mut self,
        lf: &LazyFrame,
        deterministic_rules: &[BlockingRule],
        recall: f64,
    ) -> Result<f64> {
        let lambda = estimate_lambda::estimate_probability_two_random_records_match(
            lf,
            deterministic_rules,
            &self.settings.link_type,
            &self.settings.unique_id_column,
            self.settings.source_dataset_column.as_deref(),
            recall,
        )?;
        self.settings.probability_two_random_records_match = lambda;
        Ok(lambda)
    }

    /// Estimate u-probabilities from random record pairs.
    ///
    /// Samples up to `max_pairs` random record pairs and computes the
    /// frequency of each comparison level. Since random pairs are
    /// overwhelmingly non-matches, these frequencies estimate u-probabilities.
    ///
    /// # Errors
    ///
    /// Returns [`WeldrsError::Training`]
    /// if the DataFrame has fewer than 2 records.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use weldrs::prelude::*;
    /// # let mut linker: Linker = todo!();
    /// # use polars::prelude::IntoLazy;
    /// # let lf = polars::prelude::DataFrame::empty().lazy();
    /// linker.estimate_u_using_random_sampling(&lf, 1_000_000).unwrap();
    /// ```
    pub fn estimate_u_using_random_sampling(
        &mut self,
        lf: &LazyFrame,
        max_pairs: usize,
    ) -> Result<()> {
        self.estimate_u_using_random_sampling_with_seed(lf, max_pairs, Some(42))
    }

    /// Like [`estimate_u_using_random_sampling`](Self::estimate_u_using_random_sampling)
    /// but with a configurable random seed. Pass `None` for non-deterministic sampling.
    pub fn estimate_u_using_random_sampling_with_seed(
        &mut self,
        lf: &LazyFrame,
        max_pairs: usize,
        seed: Option<u64>,
    ) -> Result<()> {
        estimate_u::estimate_u_using_random_sampling(
            lf,
            &mut self.settings.comparisons,
            max_pairs,
            &self.settings.gamma_prefix,
            &self.settings.unique_id_column,
            seed,
        )
    }

    /// Train m/u parameters using the EM algorithm, blocking on the given rule.
    ///
    /// Comparisons whose input columns overlap with the blocking rule columns
    /// are excluded from EM estimation (they always agree under that block).
    ///
    /// # Errors
    ///
    /// Returns an error if blocking, comparison-vector computation, or EM
    /// fails (typically due to Polars engine errors or missing columns).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use weldrs::prelude::*;
    /// # let mut linker: Linker = todo!();
    /// # use polars::prelude::IntoLazy;
    /// # let lf = polars::prelude::DataFrame::empty().lazy();
    /// // Train using pairs that share a last_name
    /// linker.estimate_parameters_using_em(
    ///     &lf,
    ///     &BlockingRule::on(&["last_name"]),
    /// ).unwrap();
    /// ```
    pub fn estimate_parameters_using_em(
        &mut self,
        lf: &LazyFrame,
        blocking_rule: &BlockingRule,
    ) -> Result<em::EmOutcome> {
        self.estimate_parameters_using_em_with_options(
            lf,
            blocking_rule,
            &em::EmRunOptions::default(),
        )
    }

    /// Train m/u parameters using the EM algorithm with explicit run options.
    ///
    /// Like [`estimate_parameters_using_em`](Self::estimate_parameters_using_em)
    /// but lets you control which parameters are held fixed and whether the
    /// EM-trained lambda is written back into the model (see
    /// [`EmRunOptions`](crate::em::EmRunOptions)). The no-argument variant uses
    /// [`EmRunOptions::default`], which matches Splink's defaults
    /// (`fix_u_probabilities = true`).
    ///
    /// Returns the [`EmOutcome`](crate::em::EmOutcome) (final parameters and,
    /// if `store_history` is enabled, the per-iteration trajectory) in addition
    /// to updating the model in place.
    ///
    /// # Errors
    ///
    /// Returns an error if blocking, comparison-vector computation, or EM
    /// fails (typically due to Polars engine errors or missing columns).
    pub fn estimate_parameters_using_em_with_options(
        &mut self,
        lf: &LazyFrame,
        blocking_rule: &BlockingRule,
        opts: &em::EmRunOptions,
    ) -> Result<em::EmOutcome> {
        // Generate blocked pairs using the training blocking rule.
        let blocked = blocking::generate_blocked_pairs(
            lf,
            std::slice::from_ref(blocking_rule),
            &self.settings.link_type,
            &self.settings.unique_id_column,
            self.settings.source_dataset_column.as_deref(),
        )?;

        // Compute comparison vectors.
        let cv = comparison_vectors::compute_comparison_vectors(
            blocked,
            &self.settings.comparisons,
            &self.settings.gamma_prefix,
        )?;

        // Run EM. Columns that overlap with the blocking rule have their
        // m-probabilities fixed (the column always agrees under the block).
        let columns_to_fix = blocking_rule.columns.clone();

        let outcome = em::expectation_maximization(
            &cv,
            self.settings.comparisons.clone(),
            self.settings.probability_two_random_records_match,
            &self.settings.training,
            &self.settings.gamma_prefix,
            &columns_to_fix,
            opts,
        )?;

        // Apply the final EM result back into settings. Update only the
        // comparisons whose parameters were estimated (not fixed).
        let final_result = &outcome.final_result;
        for (i, updated_comp) in final_result.comparisons.iter().enumerate() {
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

        // Optionally adopt the EM-trained lambda (off by default, matching
        // Splink's `populate_probability_two_random_records_match_from_trained_values`).
        if opts.populate_probability_two_random_records_match_from_trained_values
            && !opts.fix_probability_two_random_records_match
        {
            self.settings.probability_two_random_records_match = final_result.lambda;
        }

        Ok(outcome)
    }

    // ── Inference ─────────────────────────────────────────────────────

    /// Score record pairs using the trained model.
    ///
    /// Generates blocked pairs, computes comparison vectors, and applies the
    /// Fellegi-Sunter scoring model. Returns a `LazyFrame` with
    /// `match_weight` and `match_probability` columns.
    ///
    /// Pass `threshold_match_weight` to filter out low-scoring pairs early.
    ///
    /// # Errors
    ///
    /// Returns an error if blocking or scoring fails (e.g., missing columns
    /// or Polars engine errors).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use weldrs::prelude::*;
    /// # let linker: Linker = todo!();
    /// # use polars::prelude::IntoLazy;
    /// # let lf = polars::prelude::DataFrame::empty().lazy();
    /// // Score all pairs (no threshold)
    /// let predictions = linker.predict(&lf, None).unwrap().collect().unwrap();
    ///
    /// // Score only pairs with match weight >= 0.0
    /// let high = linker.predict(&lf, Some(0.0)).unwrap().collect().unwrap();
    /// ```
    pub fn predict(
        &self,
        lf: &LazyFrame,
        threshold_match_weight: Option<f64>,
    ) -> Result<LazyFrame> {
        self.predict_with_mode(lf, threshold_match_weight, predict::PredictMode::Auto)
    }

    /// Score record pairs using the trained model with a selectable execution strategy.
    ///
    /// `mode=Auto` chooses an implementation based on candidate-pair volume and
    /// number of comparisons.
    ///
    /// # Errors
    ///
    /// Returns an error if blocking or scoring fails (e.g., missing columns
    /// or Polars engine errors).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use weldrs::prelude::*;
    /// # let linker: Linker = todo!();
    /// # use polars::prelude::IntoLazy;
    /// # let lf = polars::prelude::DataFrame::empty().lazy();
    /// let predictions = linker
    ///     .predict_with_mode(&lf, None, PredictMode::Direct)
    ///     .unwrap()
    ///     .collect()
    ///     .unwrap();
    /// ```
    pub fn predict_with_mode(
        &self,
        lf: &LazyFrame,
        threshold_match_weight: Option<f64>,
        mode: predict::PredictMode,
    ) -> Result<LazyFrame> {
        // Attach term-frequency columns to the input before blocking so the
        // `_l` / `_r` suffixing produces `{col}_tf_l` / `{col}_tf_r`. No-op if
        // no comparison uses term-frequency adjustments.
        let lf_tf =
            crate::term_frequencies::attach_tf_columns(lf.clone(), &self.settings.comparisons)?;

        let blocked = blocking::generate_blocked_pairs(
            &lf_tf,
            &self.settings.blocking_rules,
            &self.settings.link_type,
            &self.settings.unique_id_column,
            self.settings.source_dataset_column.as_deref(),
        )?;

        let effective_mode = if mode == predict::PredictMode::Auto {
            let n_pairs = lazy_row_count(&blocked)?;
            predict::resolve_predict_mode(mode, n_pairs, self.settings.comparisons.len())
        } else {
            mode
        };

        let cv = comparison_vectors::compute_comparison_vectors(
            blocked,
            &self.settings.comparisons,
            &self.settings.gamma_prefix,
        )?;

        let scored = match effective_mode {
            predict::PredictMode::Lazy => predict::predict(
                cv,
                &self.settings.comparisons,
                self.settings.probability_two_random_records_match,
                &self.settings.gamma_prefix,
                &self.settings.bf_prefix,
                &self.settings.tf_adjustment_column_prefix,
                None,
                threshold_match_weight,
            )?,
            predict::PredictMode::Direct => {
                let cv_df = cv.collect().map_err(|e| WeldrsError::Training {
                    stage: "linker",
                    message: format!(
                        "Failed to materialize comparison vectors for direct scoring: {e}"
                    ),
                })?;
                let scored = predict::predict_direct(
                    cv_df,
                    &self.settings.comparisons,
                    self.settings.probability_two_random_records_match,
                    &self.settings.gamma_prefix,
                    &self.settings.bf_prefix,
                    &self.settings.tf_adjustment_column_prefix,
                    None,
                    threshold_match_weight,
                )?;
                scored.lazy()
            }
            predict::PredictMode::Auto => unreachable!("Auto mode should be resolved above"),
        };

        self.apply_retention(scored)
    }

    /// Project predict output down to the configured retained columns.
    ///
    /// Always keeps unique-id / source columns, `match_key`, `match_weight`,
    /// `match_probability`, and the gamma columns. `retain_matching_columns`
    /// keeps the original `{col}_l` / `{col}_r` comparison columns;
    /// `retain_intermediate_calculation_columns` keeps `bf_*` / `tf_*` columns;
    /// `additional_columns_to_retain` keeps extra payload columns.
    fn apply_retention(&self, mut lf: LazyFrame) -> Result<LazyFrame> {
        let s = &self.settings;
        let schema = lf.collect_schema().map_err(WeldrsError::Polars)?;

        let mut allowed: HashSet<String> = HashSet::new();
        let add = |name: String, allowed: &mut HashSet<String>| {
            allowed.insert(name);
        };
        let suffixed = |c: &str, allowed: &mut HashSet<String>| {
            allowed.insert(format!("{c}_l"));
            allowed.insert(format!("{c}_r"));
        };

        suffixed(&s.unique_id_column, &mut allowed);
        if let Some(src) = &s.source_dataset_column {
            suffixed(src, &mut allowed);
        }
        for name in ["match_key", "match_weight", "match_probability"] {
            add(name.to_string(), &mut allowed);
        }
        for comp in &s.comparisons {
            add(comp.gamma_column_name(&s.gamma_prefix), &mut allowed);
        }
        if s.retain_matching_columns {
            for comp in &s.comparisons {
                for ic in &comp.input_columns {
                    suffixed(ic, &mut allowed);
                }
            }
        }
        for c in &s.additional_columns_to_retain {
            suffixed(c, &mut allowed);
        }
        if s.retain_intermediate_calculation_columns {
            for comp in &s.comparisons {
                add(comp.bf_column_name(&s.bf_prefix), &mut allowed);
                add(
                    comp.tf_column_name(&s.tf_adjustment_column_prefix),
                    &mut allowed,
                );
                if comp.term_frequency_adjustments {
                    for ic in &comp.input_columns {
                        suffixed(&format!("{ic}_tf"), &mut allowed);
                    }
                }
            }
        }

        // Select, in original schema order, the columns we keep.
        let keep: Vec<Expr> = schema
            .iter_names()
            .filter(|n| allowed.contains(n.as_str()))
            .map(|n| col(n.as_str()))
            .collect();
        Ok(lf.select(keep))
    }

    /// Link records using only blocking rules, with no probabilistic scoring.
    ///
    /// Returns every candidate pair produced by the given `deterministic_rules`,
    /// annotated with `match_probability = 1.0`. Useful for high-confidence
    /// deterministic matching (the analog of Splink's `deterministic_link`).
    ///
    /// # Errors
    ///
    /// Returns an error if blocking fails (e.g. no rules, or a Polars error).
    pub fn deterministic_link(
        &self,
        lf: &LazyFrame,
        deterministic_rules: &[BlockingRule],
    ) -> Result<DataFrame> {
        let blocked = blocking::generate_blocked_pairs(
            lf,
            deterministic_rules,
            &self.settings.link_type,
            &self.settings.unique_id_column,
            self.settings.source_dataset_column.as_deref(),
        )?;
        blocked
            .with_column(lit(1.0_f64).alias("match_probability"))
            .collect()
            .map_err(WeldrsError::Polars)
    }

    /// Score a single pair of records ad hoc, bypassing blocking.
    ///
    /// Each record is a slice of `(column_name, value)` pairs. The two records
    /// are compared directly (no blocking rules applied) and scored with the
    /// trained model, returning a one-row DataFrame with the gamma, Bayes
    /// factor, `match_weight`, and `match_probability` columns — suitable for
    /// inspection or [`explain_pair`](Self::explain_pair).
    ///
    /// Note: term-frequency adjustments are **not** applied here (there is no
    /// population to derive frequencies from); scoring uses the model's m/u
    /// parameters only.
    ///
    /// # Errors
    ///
    /// Returns an error if a value cannot be converted to a Series, or if
    /// comparison-vector computation or scoring fails.
    pub fn compare_two_records(
        &self,
        record_l: &[(&str, AnyValue)],
        record_r: &[(&str, AnyValue)],
    ) -> Result<DataFrame> {
        let mut columns: Vec<Column> = Vec::with_capacity(record_l.len() + record_r.len() + 2);

        for (name, val) in record_l {
            let s = Series::from_any_values(
                format!("{name}_l").into(),
                std::slice::from_ref(val),
                true,
            )
            .map_err(WeldrsError::Polars)?;
            columns.push(s.into_column());
        }
        for (name, val) in record_r {
            let s = Series::from_any_values(
                format!("{name}_r").into(),
                std::slice::from_ref(val),
                true,
            )
            .map_err(WeldrsError::Polars)?;
            columns.push(s.into_column());
        }

        // Ensure synthetic unique-id columns exist so downstream output is
        // well-formed even if the caller didn't supply them.
        let uid_l = format!("{}_l", self.settings.unique_id_column);
        let uid_r = format!("{}_r", self.settings.unique_id_column);
        if !columns.iter().any(|c| c.name().as_str() == uid_l) {
            columns.push(Column::new(uid_l.into(), &[0i64]));
        }
        if !columns.iter().any(|c| c.name().as_str() == uid_r) {
            columns.push(Column::new(uid_r.into(), &[1i64]));
        }

        let df = DataFrame::new(1, columns).map_err(WeldrsError::Polars)?;

        // Disable term-frequency adjustments for the ad-hoc comparison.
        let comps: Vec<_> = self
            .settings
            .comparisons
            .iter()
            .cloned()
            .map(|mut c| {
                c.term_frequency_adjustments = false;
                c
            })
            .collect();

        let cv = comparison_vectors::compute_comparison_vectors(
            df.lazy(),
            &comps,
            &self.settings.gamma_prefix,
        )?;
        let cv_df = cv.collect().map_err(WeldrsError::Polars)?;
        predict::predict_direct(
            cv_df,
            &comps,
            self.settings.probability_two_random_records_match,
            &self.settings.gamma_prefix,
            &self.settings.bf_prefix,
            &self.settings.tf_adjustment_column_prefix,
            None,
            None,
        )
    }

    /// Score new records against an existing dataset without retraining.
    ///
    /// Blocks each record in `lf_new` against `lf_existing` using the model's
    /// blocking rules and scores the resulting pairs. Term frequencies (if any
    /// comparison uses them) are computed from `lf_existing` — the reference
    /// population — and applied to both sides. Most useful with a model loaded
    /// via [`load_settings_json`](Self::load_settings_json).
    ///
    /// # Errors
    ///
    /// Returns an error if blocking, comparison-vector computation, or scoring
    /// fails.
    pub fn find_matches_to_new_records(
        &self,
        lf_existing: &LazyFrame,
        lf_new: &LazyFrame,
        threshold_match_weight: Option<f64>,
    ) -> Result<LazyFrame> {
        // Term frequencies come from the existing population, applied to both
        // sides so scoring is consistent with a model trained on `lf_existing`.
        let left = crate::term_frequencies::attach_tf_columns_from(
            lf_existing.clone(),
            lf_existing,
            &self.settings.comparisons,
        )?;
        let right = crate::term_frequencies::attach_tf_columns_from(
            lf_new.clone(),
            lf_existing,
            &self.settings.comparisons,
        )?;

        let blocked = blocking::generate_blocked_pairs_between(
            &left,
            &right,
            &self.settings.blocking_rules,
            &self.settings.link_type,
            &self.settings.unique_id_column,
            self.settings.source_dataset_column.as_deref(),
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
            &self.settings.tf_adjustment_column_prefix,
            None,
            threshold_match_weight,
        )
    }

    // ── Clustering ────────────────────────────────────────────────────

    /// Cluster pairwise predictions into groups of linked records.
    ///
    /// Only pairs with `match_probability >= threshold` are considered edges.
    /// Returns a DataFrame with `[unique_id, cluster_id]` columns.
    ///
    /// # Errors
    ///
    /// Returns an error if the predictions DataFrame is missing required
    /// columns (`unique_id_l`, `unique_id_r`, `match_probability`).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use weldrs::prelude::*;
    /// # let linker: Linker = todo!();
    /// # let predictions = polars::prelude::DataFrame::empty();
    /// let clusters = linker.cluster_pairwise_predictions(&predictions, 0.5).unwrap();
    /// println!("{clusters}");
    /// ```
    pub fn cluster_pairwise_predictions(
        &self,
        predictions: &DataFrame,
        threshold: f64,
    ) -> Result<DataFrame> {
        let uid_l = format!("{}_l", self.settings.unique_id_column);
        let uid_r = format!("{}_r", self.settings.unique_id_column);
        clustering::cluster_pairwise_predictions(predictions, threshold, &uid_l, &uid_r)
    }

    /// Cluster predictions with a per-source cardinality constraint (at most one
    /// record per source dataset per cluster). Requires a `source_dataset_column`.
    ///
    /// # Errors
    ///
    /// Returns [`WeldrsError::Config`] if no source-dataset column is configured,
    /// or an error if required columns are missing.
    pub fn cluster_using_single_best_links(
        &self,
        predictions: &DataFrame,
        threshold: f64,
    ) -> Result<DataFrame> {
        let src = self
            .settings
            .source_dataset_column
            .as_deref()
            .ok_or_else(|| {
                WeldrsError::Config(
                    "cluster_using_single_best_links requires a source_dataset_column".into(),
                )
            })?;
        let uid_l = format!("{}_l", self.settings.unique_id_column);
        let uid_r = format!("{}_r", self.settings.unique_id_column);
        let src_l = format!("{src}_l");
        let src_r = format!("{src}_r");
        clustering::cluster_using_single_best_links(
            predictions,
            threshold,
            &uid_l,
            &uid_r,
            &src_l,
            &src_r,
        )
    }

    /// Compute node-, edge-, and cluster-level graph metrics for the prediction
    /// graph (degree, bridge edges, cluster density).
    ///
    /// # Errors
    ///
    /// Returns an error if required columns are missing.
    pub fn compute_graph_metrics(
        &self,
        predictions: &DataFrame,
        threshold: f64,
    ) -> Result<crate::graph_metrics::GraphMetrics> {
        let uid_l = format!("{}_l", self.settings.unique_id_column);
        let uid_r = format!("{}_r", self.settings.unique_id_column);
        crate::graph_metrics::compute_graph_metrics(predictions, threshold, &uid_l, &uid_r)
    }

    // ── Explanation ────────────────────────────────────────────────────

    /// Explain why a single record pair received its score.
    ///
    /// Returns a [`WaterfallChart`](crate::explain::WaterfallChart)
    /// showing the prior and each comparison's Bayes factor contribution.
    ///
    /// # Errors
    ///
    /// Returns an error if `row_index` is out of bounds or if required
    /// columns are missing from the predictions DataFrame.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use weldrs::prelude::*;
    /// # let linker: Linker = todo!();
    /// # let predictions = polars::prelude::DataFrame::empty();
    /// let chart = linker.explain_pair(&predictions, 0).unwrap();
    /// println!("Final probability: {}", chart.final_match_probability);
    /// for step in &chart.steps {
    ///     println!("  {} → weight {:+.2}", step.column_name, step.log2_bayes_factor);
    /// }
    /// ```
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
            &self.settings.tf_adjustment_column_prefix,
            &self.settings.unique_id_column,
        )
    }

    /// Explain why multiple record pairs received their scores.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use weldrs::prelude::*;
    /// # let linker: Linker = todo!();
    /// # let predictions = polars::prelude::DataFrame::empty();
    /// let charts = linker.explain_pairs(&predictions, &[0, 1, 2]).unwrap();
    /// assert_eq!(charts.len(), 3);
    /// ```
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
            &self.settings.tf_adjustment_column_prefix,
            &self.settings.unique_id_column,
        )
    }

    /// Produce a structured summary of the trained model parameters.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use weldrs::prelude::*;
    /// # let linker: Linker = todo!();
    /// let summary = linker.model_summary();
    /// println!("Lambda: {}", summary.probability_two_random_records_match);
    /// for comp in &summary.comparisons {
    ///     println!("Comparison: {}", comp.output_column_name);
    /// }
    /// ```
    pub fn model_summary(&self) -> explain::ModelSummary {
        explain::model_summary(&self.settings)
    }

    // ── Blocking analysis ──────────────────────────────────────────────

    /// Count the candidate pairs a single blocking rule would generate.
    ///
    /// # Errors
    ///
    /// Returns an error if blocking or the count query fails.
    pub fn count_comparisons_from_blocking_rule(
        &self,
        lf: &LazyFrame,
        rule: &BlockingRule,
    ) -> Result<u64> {
        blocking_analysis::count_comparisons_from_blocking_rule(
            lf,
            rule,
            &self.settings.link_type,
            &self.settings.unique_id_column,
            self.settings.source_dataset_column.as_deref(),
        )
    }

    /// Report the new and cumulative candidate-pair counts as blocking rules are
    /// combined (de-duplicating against earlier rules).
    ///
    /// # Errors
    ///
    /// Returns an error if blocking or a Polars op fails.
    pub fn cumulative_comparisons_from_blocking_rules(
        &self,
        lf: &LazyFrame,
        rules: &[BlockingRule],
    ) -> Result<DataFrame> {
        blocking_analysis::cumulative_comparisons_from_blocking_rules(
            lf,
            rules,
            &self.settings.link_type,
            &self.settings.unique_id_column,
            self.settings.source_dataset_column.as_deref(),
        )
    }

    /// Report the `n` largest blocks produced by an equi-join blocking rule.
    ///
    /// # Errors
    ///
    /// Returns an error if the rule has no columns or a Polars op fails.
    pub fn n_largest_blocks(
        &self,
        lf: &LazyFrame,
        rule: &BlockingRule,
        n: usize,
    ) -> Result<DataFrame> {
        blocking_analysis::n_largest_blocks(lf, rule, n)
    }

    // ── Evaluation & labelled training ─────────────────────────────────

    /// Estimate m-probabilities from an in-record ground-truth label column
    /// (e.g. a known cluster id). Records sharing a label are treated as true
    /// matches. Updates the model in place.
    ///
    /// # Errors
    ///
    /// Returns an error if blocking or comparison-vector computation fails.
    pub fn estimate_m_from_label_column(
        &mut self,
        lf: &LazyFrame,
        label_column: &str,
    ) -> Result<()> {
        evaluation::estimate_m_from_label_column(
            lf,
            &mut self.settings.comparisons,
            label_column,
            &self.settings.link_type,
            &self.settings.unique_id_column,
            self.settings.source_dataset_column.as_deref(),
            &self.settings.gamma_prefix,
        )
    }

    /// Estimate m-probabilities from an explicit table of labelled pairs
    /// (`{unique_id}_l`, `{unique_id}_r`, `is_match`). Updates the model in place.
    ///
    /// # Errors
    ///
    /// Returns an error if the labels table is malformed or a Polars op fails.
    pub fn estimate_m_from_pairwise_labels(
        &mut self,
        lf: &LazyFrame,
        labels: &DataFrame,
    ) -> Result<()> {
        evaluation::estimate_m_from_pairwise_labels(
            lf,
            labels,
            &mut self.settings.comparisons,
            &self.settings.unique_id_column,
            &self.settings.gamma_prefix,
        )
    }

    /// Confusion-matrix metrics across a threshold sweep, given scored
    /// predictions and a labelled-pair table.
    ///
    /// # Errors
    ///
    /// Returns an error if required columns are missing.
    pub fn accuracy_analysis(
        &self,
        predictions: &DataFrame,
        labels: &DataFrame,
    ) -> Result<Vec<evaluation::ThresholdMetrics>> {
        evaluation::accuracy_analysis(predictions, labels, &self.settings.unique_id_column)
    }

    /// ROC table (`[threshold, fpr, tpr]`) from predictions and labels.
    ///
    /// # Errors
    ///
    /// Returns an error if metric computation fails.
    pub fn roc_table(&self, predictions: &DataFrame, labels: &DataFrame) -> Result<DataFrame> {
        evaluation::roc_table(predictions, labels, &self.settings.unique_id_column)
    }

    /// Precision-recall table (`[threshold, precision, recall]`).
    ///
    /// # Errors
    ///
    /// Returns an error if metric computation fails.
    pub fn precision_recall_table(
        &self,
        predictions: &DataFrame,
        labels: &DataFrame,
    ) -> Result<DataFrame> {
        evaluation::precision_recall_table(predictions, labels, &self.settings.unique_id_column)
    }

    /// False positives and false negatives at a given threshold.
    ///
    /// # Errors
    ///
    /// Returns an error if required columns are missing.
    pub fn prediction_errors_from_labels(
        &self,
        predictions: &DataFrame,
        labels: &DataFrame,
        threshold: f64,
    ) -> Result<DataFrame> {
        evaluation::prediction_errors_from_labels(
            predictions,
            labels,
            &self.settings.unique_id_column,
            threshold,
        )
    }

    // ── Serialization ─────────────────────────────────────────────────

    /// Serialize the current settings (including trained parameters) to JSON.
    ///
    /// # Errors
    ///
    /// Returns [`WeldrsError::Serde`] if
    /// serialization fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use weldrs::prelude::*;
    /// # let linker: Linker = todo!();
    /// let json = linker.save_settings_json().unwrap();
    /// std::fs::write("model.json", &json).unwrap();
    /// ```
    pub fn save_settings_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.settings).map_err(Into::into)
    }

    /// Load a linker from previously saved JSON settings.
    ///
    /// # Errors
    ///
    /// Returns [`WeldrsError::Serde`] if
    /// the JSON is malformed or does not match the expected schema.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use weldrs::prelude::*;
    /// let json = std::fs::read_to_string("model.json").unwrap();
    /// let linker = Linker::load_settings_json(&json).unwrap();
    /// ```
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

    /// Render an m/u parameters chart for the trained model as an SVG string.
    pub fn m_u_parameters_chart_svg(
        &self,
        options: &crate::visualize::ChartOptions,
    ) -> Result<String> {
        let summary = self.model_summary();
        crate::visualize::m_u_parameters_chart_svg(&summary, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predict::PredictMode;
    use crate::settings::LinkType;
    use crate::test_helpers;

    fn make_linker() -> Linker {
        let settings = crate::settings::Settings::builder(LinkType::DedupeOnly)
            .comparison(test_helpers::exact_match_comparison("first_name"))
            .comparison(test_helpers::exact_match_comparison("last_name"))
            .blocking_rule(BlockingRule::on(&["last_name"]))
            .build()
            .unwrap();
        Linker::new(settings).unwrap()
    }

    #[test]
    fn test_linker_new() {
        let linker = make_linker();
        assert_eq!(linker.settings().comparisons.len(), 2);
        assert_eq!(linker.settings().link_type, LinkType::DedupeOnly);
    }

    #[test]
    fn test_save_load_settings_json() {
        let linker = make_linker();
        let json = linker.save_settings_json().unwrap();
        let restored = Linker::load_settings_json(&json).unwrap();

        assert_eq!(
            restored.settings().comparisons.len(),
            linker.settings().comparisons.len()
        );
        assert_eq!(restored.settings().link_type, linker.settings().link_type);
        assert_eq!(
            restored.settings().unique_id_column,
            linker.settings().unique_id_column
        );
    }

    #[test]
    fn test_estimate_lambda_updates_settings() {
        let mut linker = make_linker();
        let lf = test_helpers::make_test_df().lazy();

        let initial_lambda = linker.settings().probability_two_random_records_match;

        linker
            .estimate_probability_two_random_records_match(
                &lf,
                &[BlockingRule::on(&["first_name", "last_name"])],
                1.0,
            )
            .unwrap();

        assert_ne!(
            linker.settings().probability_two_random_records_match,
            initial_lambda,
            "Lambda should change after estimation"
        );
    }

    #[test]
    fn test_predict_returns_scored_pairs() {
        let mut linker = make_linker();
        let lf = test_helpers::make_test_df().lazy();

        // Estimate u values so BFs are reasonable
        linker.estimate_u_using_random_sampling(&lf, 100).unwrap();

        let predictions = linker.predict(&lf, None).unwrap().collect().unwrap();

        let col_names: Vec<&str> = predictions
            .get_column_names()
            .into_iter()
            .map(|s| s.as_str())
            .collect();
        assert!(col_names.contains(&"match_weight"));
        assert!(col_names.contains(&"match_probability"));
        assert!(predictions.height() > 0);
    }

    #[test]
    fn test_predict_with_mode_direct_matches_lazy() {
        let mut linker = make_linker();
        let lf = test_helpers::make_test_df().lazy();

        linker.estimate_u_using_random_sampling(&lf, 100).unwrap();

        let lazy = linker
            .predict_with_mode(&lf, None, PredictMode::Lazy)
            .unwrap()
            .collect()
            .unwrap();
        let direct = linker
            .predict_with_mode(&lf, None, PredictMode::Direct)
            .unwrap()
            .collect()
            .unwrap();

        let lazy_probs: Vec<f64> = lazy
            .column("match_probability")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        let direct_probs: Vec<f64> = direct
            .column("match_probability")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();

        assert_eq!(lazy_probs.len(), direct_probs.len());
        for (lp, dp) in lazy_probs.iter().zip(direct_probs.iter()) {
            assert!((lp - dp).abs() < 1e-10, "Mismatch: {lp} vs {dp}");
        }
    }

    #[test]
    fn test_em_does_not_change_lambda_by_default() {
        let mut linker = make_linker();
        let lf = test_helpers::make_test_df().lazy();
        linker.estimate_u_using_random_sampling(&lf, 100).unwrap();

        let before = linker.settings().probability_two_random_records_match;
        linker
            .estimate_parameters_using_em(&lf, &BlockingRule::on(&["last_name"]))
            .unwrap();
        let after = linker.settings().probability_two_random_records_match;

        assert_eq!(
            before, after,
            "lambda should be untouched by EM unless populate flag is set"
        );
    }

    #[test]
    fn test_em_populates_lambda_when_requested() {
        use crate::em::EmRunOptions;
        let mut linker = make_linker();
        let lf = test_helpers::make_test_df().lazy();
        linker.estimate_u_using_random_sampling(&lf, 100).unwrap();

        let before = linker.settings().probability_two_random_records_match;
        let opts = EmRunOptions {
            populate_probability_two_random_records_match_from_trained_values: true,
            ..Default::default()
        };
        linker
            .estimate_parameters_using_em_with_options(
                &lf,
                &BlockingRule::on(&["last_name"]),
                &opts,
            )
            .unwrap();
        let after = linker.settings().probability_two_random_records_match;

        assert_ne!(
            before, after,
            "lambda should be adopted from EM when the populate flag is set"
        );
    }

    #[test]
    fn test_predict_with_mode_auto() {
        let mut linker = make_linker();
        let lf = test_helpers::make_test_df().lazy();

        linker.estimate_u_using_random_sampling(&lf, 100).unwrap();

        let predictions = linker
            .predict_with_mode(&lf, None, PredictMode::Auto)
            .unwrap()
            .collect()
            .unwrap();
        assert!(predictions.height() > 0);
        assert!(predictions.column("match_weight").is_ok());
        assert!(predictions.column("match_probability").is_ok());
    }
}
