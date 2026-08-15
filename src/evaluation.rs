//! Model evaluation and labelled-data training.
//!
//! This module covers the supervised side of record linkage:
//!
//! - **m-estimation from labels** — when ground-truth matches are known, the
//!   m-probabilities can be estimated directly instead of (or in addition to)
//!   via EM. [`estimate_m_from_label_column`] uses an in-record ground-truth id
//!   column; [`estimate_m_from_pairwise_labels`] uses an explicit table of
//!   labelled pairs.
//! - **accuracy metrics** — given scored predictions and a labelled set of
//!   pairs, [`accuracy_analysis`], [`roc_table`], [`precision_recall_table`],
//!   and [`prediction_errors_from_labels`] quantify model quality.
//!
//! Labelled-pair tables are expected to contain `{unique_id}_l` and
//! `{unique_id}_r` columns plus a boolean `is_match` column.

use polars::prelude::*;

use crate::blocking::{self, BlockingRule};
use crate::comparison::Comparison;
use crate::comparison_vectors::compute_comparison_vectors;
use crate::error::{Result, WeldrsError};
use crate::settings::LinkType;
use crate::training_common::{ProbKind, assign_level_frequencies, group_agreement_patterns};

/// Estimate m-probabilities from an in-record ground-truth label column.
///
/// Records sharing a value in `label_column` (e.g. a known cluster id or SSN)
/// are treated as true matches. The m-probability of each comparison level is
/// its frequency of agreement among those true-match pairs. Comparisons are
/// updated in place.
///
/// # Errors
///
/// Returns an error if blocking, comparison-vector computation, or pattern
/// counting fails.
pub fn estimate_m_from_label_column(
    lf: &LazyFrame,
    comparisons: &mut [Comparison],
    label_column: &str,
    link_type: &LinkType,
    unique_id_col: &str,
    source_dataset_column: Option<&str>,
    gamma_prefix: &str,
) -> Result<()> {
    let rule = BlockingRule::on(&[label_column]);
    let blocked = blocking::generate_blocked_pairs(
        lf,
        std::slice::from_ref(&rule),
        link_type,
        unique_id_col,
        source_dataset_column,
    )?;
    let cv = compute_comparison_vectors(blocked, comparisons, gamma_prefix)?;
    let (pattern_counts, counts) =
        group_agreement_patterns(cv, comparisons, gamma_prefix, "estimate_m")?;
    assign_level_frequencies(
        &pattern_counts,
        &counts,
        comparisons,
        gamma_prefix,
        ProbKind::M,
        "estimate_m",
    )?;
    Ok(())
}

/// Estimate m-probabilities from an explicit table of labelled pairs.
///
/// `labels` must contain `{unique_id}_l` and `{unique_id}_r` columns and a
/// boolean `is_match` column. Only true-match pairs are used. Record attributes
/// are joined from `lf` onto each labelled pair, comparison vectors are
/// computed, and per-level agreement frequencies become the m-probabilities.
///
/// # Errors
///
/// Returns an error if the labels table is missing required columns or any
/// Polars operation fails.
pub fn estimate_m_from_pairwise_labels(
    lf: &LazyFrame,
    labels: &DataFrame,
    comparisons: &mut [Comparison],
    unique_id_col: &str,
    gamma_prefix: &str,
) -> Result<()> {
    let uid_l = format!("{unique_id_col}_l");
    let uid_r = format!("{unique_id_col}_r");

    // Keep only true-match labelled pairs.
    let matches = labels
        .clone()
        .lazy()
        .filter(col("is_match"))
        .select([col(uid_l.as_str()), col(uid_r.as_str())]);

    // Suffix the record attributes and join them onto each labelled pair.
    let left = lf.clone().select([col("*").name().suffix("_l")]);
    let right = lf.clone().select([col("*").name().suffix("_r")]);

    let paired = matches
        .join(
            left,
            [col(uid_l.as_str())],
            [col(uid_l.as_str())],
            JoinArgs::new(JoinType::Inner),
        )
        .join(
            right,
            [col(uid_r.as_str())],
            [col(uid_r.as_str())],
            JoinArgs::new(JoinType::Inner),
        );

    let cv = compute_comparison_vectors(paired, comparisons, gamma_prefix)?;
    let (pattern_counts, counts) =
        group_agreement_patterns(cv, comparisons, gamma_prefix, "estimate_m")?;
    assign_level_frequencies(
        &pattern_counts,
        &counts,
        comparisons,
        gamma_prefix,
        ProbKind::M,
        "estimate_m",
    )?;
    Ok(())
}

/// A single confusion-matrix row at one threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdMetrics {
    /// Match-probability threshold (pairs with `p >= threshold` are predicted matches).
    pub threshold: f64,
    /// True positives.
    pub tp: u64,
    /// False positives.
    pub fp: u64,
    /// True negatives.
    pub tn: u64,
    /// False negatives.
    pub fn_: u64,
    /// `tp / (tp + fp)` (1.0 when no predicted positives).
    pub precision: f64,
    /// `tp / (tp + fn)` (recall / true-positive rate).
    pub recall: f64,
    /// `tn / (tn + fp)` (specificity / true-negative rate).
    pub specificity: f64,
    /// Harmonic mean of precision and recall.
    pub f1: f64,
}

/// Join scored predictions to a labelled-pair table, yielding `(probability,
/// is_match)` for every labelled pair. Pairs absent from `predictions` (e.g.
/// filtered by blocking or a threshold) are treated as probability `0.0`.
fn joined_truth(
    predictions: &DataFrame,
    labels: &DataFrame,
    unique_id_col: &str,
) -> Result<Vec<(f64, bool)>> {
    let uid_l = format!("{unique_id_col}_l");
    let uid_r = format!("{unique_id_col}_r");

    let preds = predictions.clone().lazy().select([
        col(uid_l.as_str()),
        col(uid_r.as_str()),
        col("match_probability"),
    ]);

    let joined = labels
        .clone()
        .lazy()
        .join(
            preds,
            [col(uid_l.as_str()), col(uid_r.as_str())],
            [col(uid_l.as_str()), col(uid_r.as_str())],
            JoinArgs::new(JoinType::Left),
        )
        .with_column(col("match_probability").fill_null(lit(0.0)))
        .collect()
        .map_err(WeldrsError::Polars)?;

    let probs = joined
        .column("match_probability")
        .map_err(WeldrsError::Polars)?
        .f64()
        .map_err(WeldrsError::Polars)?;
    let is_match = joined
        .column("is_match")
        .map_err(WeldrsError::Polars)?
        .bool()
        .map_err(WeldrsError::Polars)?;

    Ok(probs
        .into_iter()
        .zip(is_match)
        .map(|(p, m)| (p.unwrap_or(0.0), m.unwrap_or(false)))
        .collect())
}

/// Compute confusion-matrix metrics across a sweep of thresholds.
///
/// Returns one [`ThresholdMetrics`] per distinct predicted probability (plus a
/// trivial `threshold = 0.0` point), sorted ascending by threshold. `labels`
/// defines the universe of reviewed pairs (see module docs).
///
/// # Errors
///
/// Returns an error if required columns are missing from `predictions` or
/// `labels`.
pub fn accuracy_analysis(
    predictions: &DataFrame,
    labels: &DataFrame,
    unique_id_col: &str,
) -> Result<Vec<ThresholdMetrics>> {
    let truth = joined_truth(predictions, labels, unique_id_col)?;
    let total_pos = truth.iter().filter(|(_, m)| *m).count() as u64;
    let total_neg = truth.len() as u64 - total_pos;

    // Candidate thresholds: 0.0 plus every distinct predicted probability.
    let mut thresholds: Vec<f64> = truth.iter().map(|(p, _)| *p).collect();
    thresholds.push(0.0);
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    let mut out = Vec::with_capacity(thresholds.len());
    for t in thresholds {
        let mut tp = 0u64;
        let mut fp = 0u64;
        for (p, m) in &truth {
            if *p >= t {
                if *m {
                    tp += 1;
                } else {
                    fp += 1;
                }
            }
        }
        let fn_ = total_pos - tp;
        let tn = total_neg - fp;
        let precision = if tp + fp == 0 {
            1.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if total_pos == 0 {
            0.0
        } else {
            tp as f64 / total_pos as f64
        };
        let specificity = if total_neg == 0 {
            1.0
        } else {
            tn as f64 / total_neg as f64
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        out.push(ThresholdMetrics {
            threshold: t,
            tp,
            fp,
            tn,
            fn_,
            precision,
            recall,
            specificity,
            f1,
        });
    }
    Ok(out)
}

/// Build a `[threshold, fpr, tpr]` ROC table from scored predictions and labels.
///
/// # Errors
///
/// Returns an error if metric computation fails.
pub fn roc_table(
    predictions: &DataFrame,
    labels: &DataFrame,
    unique_id_col: &str,
) -> Result<DataFrame> {
    let metrics = accuracy_analysis(predictions, labels, unique_id_col)?;
    let threshold: Vec<f64> = metrics.iter().map(|m| m.threshold).collect();
    let tpr: Vec<f64> = metrics.iter().map(|m| m.recall).collect();
    let fpr: Vec<f64> = metrics.iter().map(|m| 1.0 - m.specificity).collect();
    DataFrame::new(
        metrics.len(),
        vec![
            Column::new("threshold".into(), threshold),
            Column::new("fpr".into(), fpr),
            Column::new("tpr".into(), tpr),
        ],
    )
    .map_err(WeldrsError::Polars)
}

/// Build a `[threshold, precision, recall]` table from predictions and labels.
///
/// # Errors
///
/// Returns an error if metric computation fails.
pub fn precision_recall_table(
    predictions: &DataFrame,
    labels: &DataFrame,
    unique_id_col: &str,
) -> Result<DataFrame> {
    let metrics = accuracy_analysis(predictions, labels, unique_id_col)?;
    let threshold: Vec<f64> = metrics.iter().map(|m| m.threshold).collect();
    let precision: Vec<f64> = metrics.iter().map(|m| m.precision).collect();
    let recall: Vec<f64> = metrics.iter().map(|m| m.recall).collect();
    DataFrame::new(
        metrics.len(),
        vec![
            Column::new("threshold".into(), threshold),
            Column::new("precision".into(), precision),
            Column::new("recall".into(), recall),
        ],
    )
    .map_err(WeldrsError::Polars)
}

/// Return the labelled pairs the model gets wrong at `threshold`: false
/// positives (predicted match, labelled non-match) and false negatives
/// (predicted non-match, labelled match).
///
/// The result contains the label columns plus `match_probability` and an
/// `error_type` column (`"false_positive"` / `"false_negative"`).
///
/// # Errors
///
/// Returns an error if required columns are missing or a Polars op fails.
pub fn prediction_errors_from_labels(
    predictions: &DataFrame,
    labels: &DataFrame,
    unique_id_col: &str,
    threshold: f64,
) -> Result<DataFrame> {
    let uid_l = format!("{unique_id_col}_l");
    let uid_r = format!("{unique_id_col}_r");

    let preds = predictions.clone().lazy().select([
        col(uid_l.as_str()),
        col(uid_r.as_str()),
        col("match_probability"),
    ]);

    labels
        .clone()
        .lazy()
        .join(
            preds,
            [col(uid_l.as_str()), col(uid_r.as_str())],
            [col(uid_l.as_str()), col(uid_r.as_str())],
            JoinArgs::new(JoinType::Left),
        )
        .with_column(col("match_probability").fill_null(lit(0.0)))
        // Keep only disagreements between prediction and label.
        .filter(
            col("match_probability")
                .gt_eq(lit(threshold))
                .neq(col("is_match")),
        )
        .with_column(
            when(col("match_probability").gt_eq(lit(threshold)))
                .then(lit("false_positive"))
                .otherwise(lit("false_negative"))
                .alias("error_type"),
        )
        .collect()
        .map_err(WeldrsError::Polars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison::ComparisonBuilder;

    fn labelled_records() -> LazyFrame {
        df!(
            "unique_id" => [1i64, 2, 3, 4],
            // ground-truth cluster: {1,2} are the same entity, 3 and 4 distinct.
            "entity" => [10i64, 10, 20, 30],
            "first_name" => ["John", "Jon", "Mary", "Sue"],
        )
        .unwrap()
        .lazy()
    }

    #[test]
    fn test_estimate_m_from_label_column_sets_m() {
        let mut comps = vec![
            ComparisonBuilder::new("first_name")
                .null_level()
                .exact_match_level()
                .jaro_winkler_level(0.8)
                .else_level()
                .build()
                .unwrap(),
        ];

        estimate_m_from_label_column(
            &labelled_records(),
            &mut comps,
            "entity",
            &LinkType::DedupeOnly,
            "unique_id",
            None,
            "gamma_",
        )
        .unwrap();

        // The only true-match pair is (1,2): "John"/"Jon" → Jaro-Winkler level.
        // So m for the JW level should dominate.
        let jw = comps[0]
            .comparison_levels
            .iter()
            .find(|l| l.comparison_vector_value == 1)
            .unwrap();
        assert_eq!(jw.m_probability, Some(1.0));
    }

    #[test]
    fn test_estimate_m_from_pairwise_labels() {
        let labels = df!(
            "unique_id_l" => [1i64],
            "unique_id_r" => [2i64],
            "is_match" => [true],
        )
        .unwrap();

        let mut comps = vec![
            ComparisonBuilder::new("first_name")
                .null_level()
                .exact_match_level()
                .jaro_winkler_level(0.8)
                .else_level()
                .build()
                .unwrap(),
        ];

        estimate_m_from_pairwise_labels(
            &labelled_records(),
            &labels,
            &mut comps,
            "unique_id",
            "gamma_",
        )
        .unwrap();

        let jw = comps[0]
            .comparison_levels
            .iter()
            .find(|l| l.comparison_vector_value == 1)
            .unwrap();
        assert_eq!(jw.m_probability, Some(1.0));
    }

    fn predictions_and_labels() -> (DataFrame, DataFrame) {
        // 4 labelled pairs; predictions assign probabilities.
        let predictions = df!(
            "unique_id_l" => [1i64, 1, 2, 3],
            "unique_id_r" => [2i64, 3, 3, 4],
            "match_probability" => [0.95, 0.10, 0.80, 0.40],
        )
        .unwrap();
        let labels = df!(
            "unique_id_l" => [1i64, 1, 2, 3],
            "unique_id_r" => [2i64, 3, 3, 4],
            "is_match" => [true, false, true, false],
        )
        .unwrap();
        (predictions, labels)
    }

    #[test]
    fn test_accuracy_analysis_perfect_threshold() {
        let (predictions, labels) = predictions_and_labels();
        let metrics = accuracy_analysis(&predictions, &labels, "unique_id").unwrap();

        // At threshold 0.80: matches (1,2)=0.95 and (2,3)=0.80 predicted (both
        // true); non-matches (1,3)=0.10 and (3,4)=0.40 below → perfect split.
        let at_080 = metrics
            .iter()
            .find(|m| (m.threshold - 0.80).abs() < 1e-9)
            .expect("threshold 0.80 present");
        assert_eq!(at_080.tp, 2);
        assert_eq!(at_080.fp, 0);
        assert_eq!(at_080.fn_, 0);
        assert_eq!(at_080.tn, 2);
        assert!((at_080.precision - 1.0).abs() < 1e-12);
        assert!((at_080.recall - 1.0).abs() < 1e-12);
        assert!((at_080.f1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_roc_and_pr_tables_shapes() {
        let (predictions, labels) = predictions_and_labels();
        let roc = roc_table(&predictions, &labels, "unique_id").unwrap();
        assert_eq!(
            roc.get_column_names()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            vec!["threshold", "fpr", "tpr"]
        );
        let pr = precision_recall_table(&predictions, &labels, "unique_id").unwrap();
        assert!(pr.height() > 0);
        assert!(
            pr.get_column_names()
                .iter()
                .any(|s| s.as_str() == "precision")
        );
    }

    #[test]
    fn test_prediction_errors_flags_fp_and_fn() {
        // Predict everything as a strong match → (1,3) and (3,4) become FPs.
        let predictions = df!(
            "unique_id_l" => [1i64, 1, 2, 3],
            "unique_id_r" => [2i64, 3, 3, 4],
            "match_probability" => [0.95, 0.95, 0.05, 0.95],
        )
        .unwrap();
        let labels = df!(
            "unique_id_l" => [1i64, 1, 2, 3],
            "unique_id_r" => [2i64, 3, 3, 4],
            "is_match" => [true, false, true, false],
        )
        .unwrap();

        let errors =
            prediction_errors_from_labels(&predictions, &labels, "unique_id", 0.5).unwrap();
        // (1,3) FP, (3,4) FP, (2,3) FN  → 3 errors.
        assert_eq!(errors.height(), 3);
        let kinds: Vec<&str> = errors
            .column("error_type")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(kinds.iter().filter(|k| **k == "false_positive").count(), 2);
        assert_eq!(kinds.iter().filter(|k| **k == "false_negative").count(), 1);
    }
}
