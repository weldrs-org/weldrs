//! Shared helpers for parameter estimation.
//!
//! Both [`estimate_u`](crate::estimate_u) (u from random pairs) and
//! [`evaluation::estimate_m_from_label_column`](crate::evaluation::estimate_m_from_label_column)
//! (m from true-match pairs) reduce candidate pairs to agreement-pattern counts
//! and then turn per-level frequencies into probabilities. These helpers
//! capture that shared logic.

use polars::prelude::*;

use crate::comparison::Comparison;
use crate::error::{Result, WeldrsError};

/// Which probability a level frequency is assigned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProbKind {
    /// Assign to `m_probability` (frequencies among true matches).
    M,
    /// Assign to `u_probability` (frequencies among non-matches).
    U,
}

/// Group comparison vectors by their gamma columns and return the distinct
/// agreement patterns together with their counts.
///
/// Returns the grouped `DataFrame` (containing one gamma column per comparison
/// plus a `__count` column) and the counts as an aligned `Vec<f64>`.
pub(crate) fn group_agreement_patterns(
    cv: LazyFrame,
    comparisons: &[Comparison],
    gamma_prefix: &str,
    stage: &'static str,
) -> Result<(DataFrame, Vec<f64>)> {
    let group_exprs: Vec<Expr> = comparisons
        .iter()
        .map(|c| col(c.gamma_column_name(gamma_prefix)))
        .collect();

    let pattern_counts = cv
        .group_by(group_exprs)
        .agg([len().alias("__count")])
        .collect()
        .map_err(|e| WeldrsError::Training {
            stage,
            message: format!("Failed to count agreement patterns: {e}"),
        })?;

    let counts: Vec<f64> = pattern_counts
        .column("__count")
        .map_err(|e| WeldrsError::Training {
            stage,
            message: format!("Missing count column: {e}"),
        })?
        .u32()
        .map_err(|e| WeldrsError::Training {
            stage,
            message: format!("Count column type error: {e}"),
        })?
        .into_no_null_iter()
        .map(|v| v as f64)
        .collect();

    Ok((pattern_counts, counts))
}

/// Assign per-level frequencies (computed from grouped agreement patterns) to
/// each comparison's m- or u-probabilities.
///
/// For each comparison, the frequency of a non-null level is its share of all
/// non-null pairs. Null levels and levels with the relevant `fix_*` flag set
/// are left untouched.
pub(crate) fn assign_level_frequencies(
    pattern_counts: &DataFrame,
    counts: &[f64],
    comparisons: &mut [Comparison],
    gamma_prefix: &str,
    kind: ProbKind,
    stage: &'static str,
) -> Result<()> {
    for comp in comparisons.iter_mut() {
        let gamma_col_name = comp.gamma_column_name(gamma_prefix);
        let gamma_series =
            pattern_counts
                .column(&gamma_col_name)
                .map_err(|e| WeldrsError::Training {
                    stage,
                    message: format!("Missing gamma column: {e}"),
                })?;
        let gammas = gamma_series.i8().map_err(|e| WeldrsError::Training {
            stage,
            message: format!("Gamma type error: {e}"),
        })?;

        // Total weight over non-null levels for this comparison.
        let mut total_non_null = 0.0_f64;
        for (row, &count) in counts.iter().enumerate() {
            let gv = gammas.get(row).unwrap_or(-1) as i32;
            let is_null = comp
                .comparison_levels
                .iter()
                .any(|l| l.comparison_vector_value == gv && l.is_null_level);
            if !is_null {
                total_non_null += count;
            }
        }

        if total_non_null <= 0.0 {
            continue;
        }

        for level in &mut comp.comparison_levels {
            if level.is_null_level {
                continue;
            }
            match kind {
                ProbKind::M if level.fix_m_probability => continue,
                ProbKind::U if level.fix_u_probability => continue,
                _ => {}
            }

            let cv = level.comparison_vector_value as i8;
            let mut level_count = 0.0_f64;
            for (row, &count) in counts.iter().enumerate() {
                if gammas.get(row) == Some(cv) {
                    level_count += count;
                }
            }
            let prob = level_count / total_non_null;
            match kind {
                ProbKind::M => level.m_probability = Some(prob),
                ProbKind::U => level.u_probability = Some(prob),
            }
        }
    }
    Ok(())
}
