//! Expectation-Maximization (EM) training.
//!
//! The EM algorithm iteratively estimates m-probabilities, u-probabilities,
//! and lambda from agreement-pattern counts derived from blocked record
//! pairs.
//!
//! The **E-step** computes, for each unique agreement pattern, the
//! posterior probability that the pair is a true match given the current
//! parameters. The **M-step** re-estimates m, u, and lambda from these
//! posterior weights. Iterations continue until the largest parameter
//! change falls below the convergence threshold set in
//! [`TrainingSettings`].
//!
//! Most users call this via
//! [`Linker::estimate_parameters_using_em`](crate::linker::Linker::estimate_parameters_using_em)
//! rather than invoking [`expectation_maximization`] directly.
//!
//! See also: [`estimate_u`](crate::estimate_u) for u-probability
//! initialization and [`estimate_lambda`](crate::estimate_lambda) for
//! lambda initialization.

use polars::prelude::*;
use rayon::prelude::*;

use crate::comparison::Comparison;
use crate::error::{Result, WeldrsError};
use crate::probability;
use crate::settings::TrainingSettings;

/// Result of a single EM iteration.
#[derive(Debug, Clone)]
pub struct EmIterationResult {
    /// Zero-based iteration index.
    pub iteration: usize,
    /// Updated lambda (prior match probability) after this iteration.
    pub lambda: f64,
    /// Maximum absolute change in any m/u parameter during this iteration.
    /// Convergence is declared when this falls below the threshold.
    pub max_change: f64,
    /// Snapshot of all comparisons with their updated m/u probabilities.
    pub comparisons: Vec<Comparison>,
}

/// Outcome of an EM training run.
///
/// Always carries the `final_result` (the converged / last iteration). The
/// per-iteration `history` is `Some` only when
/// [`TrainingSettings::store_history`](crate::settings::TrainingSettings) is
/// enabled — useful for plotting parameter trajectories
/// (see [`parameter_estimate_comparisons_chart_svg`](crate::visualize::parameter_estimate_comparisons_chart_svg)).
#[derive(Debug, Clone)]
pub struct EmOutcome {
    /// The final iteration's parameters (converged or last before `max_iterations`).
    pub final_result: EmIterationResult,
    /// Full per-iteration history, if `store_history` was enabled.
    pub history: Option<Vec<EmIterationResult>>,
}

/// Options controlling a single EM training run.
///
/// Defaults match Splink's
/// `estimate_parameters_using_expectation_maximisation`: `fix_u_probabilities`
/// is `true` (u-probabilities come from
/// [`estimate_u`](crate::estimate_u) and are not re-estimated during EM), while
/// every other flag is `false`.
///
/// Note: the **free function** [`expectation_maximization`] takes these options
/// explicitly (no hidden default), while
/// [`Linker::estimate_parameters_using_em`](crate::linker::Linker::estimate_parameters_using_em)
/// applies [`EmRunOptions::default`] to match Splink behaviour. These two layers
/// are intentionally distinct — do not "unify" them, or the Splink-conformant
/// defaults will silently change.
#[derive(Debug, Clone)]
pub struct EmRunOptions {
    /// If `true`, m-probabilities are held fixed for all comparisons.
    pub fix_m_probabilities: bool,
    /// If `true`, u-probabilities are held fixed for all comparisons
    /// (the Splink default — u comes from random-sampling estimation).
    pub fix_u_probabilities: bool,
    /// If `true`, lambda (probability two random records match) is not updated
    /// by the M-step.
    pub fix_probability_two_random_records_match: bool,
    /// If `true`, the EM-trained lambda is written back into the model settings
    /// by [`Linker::estimate_parameters_using_em`](crate::linker::Linker::estimate_parameters_using_em).
    /// Has no effect if `fix_probability_two_random_records_match` is `true`.
    pub populate_probability_two_random_records_match_from_trained_values: bool,
}

impl Default for EmRunOptions {
    fn default() -> Self {
        Self {
            fix_m_probabilities: false,
            fix_u_probabilities: true,
            fix_probability_two_random_records_match: false,
            populate_probability_two_random_records_match_from_trained_values: false,
        }
    }
}

/// Run the EM algorithm on comparison vectors.
///
/// `comparison_vectors` must contain gamma columns for every comparison in
/// `comparisons`. Comparisons whose input columns overlap `columns_to_fix`
/// (the training blocking rule) have their **m-probabilities** held fixed —
/// the column always agrees under the block, so its m is indeterminate from
/// this pass. u-probabilities are governed solely by `opts.fix_u_probabilities`
/// (independent of block overlap), since u is estimated separately.
///
/// Returns an [`EmOutcome`] with the final parameters and (when
/// `store_history` is enabled) the full per-iteration history.
///
/// # Errors
///
/// Returns an error if gamma columns are missing from the comparison
/// vectors or if Polars group-by / aggregation fails.
pub fn expectation_maximization(
    comparison_vectors: &LazyFrame,
    mut comparisons: Vec<Comparison>,
    lambda: f64,
    training: &TrainingSettings,
    gamma_prefix: &str,
    columns_to_fix: &[String],
    opts: &EmRunOptions,
) -> Result<EmOutcome> {
    // Determine which parameters are held fixed for each comparison, and which
    // comparisons overlap the training block.
    //
    // - Block overlap fixes only m (the column always agrees under the block).
    // - `opts.fix_m_probabilities` fixes m globally.
    // - `opts.fix_u_probabilities` fixes u globally (independent of overlap).
    //
    // A block-overlapping comparison is also *excluded* from the E-step: under
    // the block it always agrees, so it carries no information about whether a
    // pair matches. Leaving it in (with its default/previous Bayes factor) would
    // saturate the posterior and bias the other comparisons' m estimates. This
    // mirrors Splink, which drops the blocked column from the EM session.
    let excluded: Vec<bool> = comparisons
        .iter()
        .map(|comp| {
            columns_to_fix
                .iter()
                .any(|c| comp.input_columns.contains(c))
        })
        .collect();
    for (comp, &overlaps) in comparisons.iter_mut().zip(excluded.iter()) {
        for level in &mut comp.comparison_levels {
            if overlaps || opts.fix_m_probabilities {
                level.fix_m_probability = true;
            }
            if opts.fix_u_probabilities {
                level.fix_u_probability = true;
            }
        }
    }

    // Step 1: Compute agreement pattern counts.
    // Group by all gamma columns to collapse millions of pairs into a few
    // hundred unique patterns.
    let gamma_cols: Vec<String> = comparisons
        .iter()
        .map(|c| c.gamma_column_name(gamma_prefix))
        .collect();

    let group_exprs: Vec<Expr> = gamma_cols.iter().map(|c| col(c.as_str())).collect();

    let pattern_counts = comparison_vectors
        .clone()
        .group_by(group_exprs)
        .agg([len().alias("__count")])
        .collect()
        .map_err(|e| WeldrsError::Training {
            stage: "em",
            message: format!("Failed to count agreement patterns: {e}"),
        })?;

    // Pre-extract gamma columns once — shared across all E/M steps.
    let gamma_columns: Vec<Vec<i8>> = comparisons
        .iter()
        .map(|comp| {
            let col_name = comp.gamma_column_name(gamma_prefix);
            let series = pattern_counts
                .column(&col_name)
                .map_err(|e| WeldrsError::Training {
                    stage: "em",
                    message: format!("Missing gamma column: {e}"),
                })?;
            let gammas = series.i8().map_err(|e| WeldrsError::Training {
                stage: "em",
                message: format!("Gamma column type error: {e}"),
            })?;
            Ok(gammas.into_iter().map(|v| v.unwrap_or(-1i8)).collect())
        })
        .collect::<Result<Vec<_>>>()?;

    // Pre-extract counts once.
    let count_series = pattern_counts
        .column("__count")
        .map_err(|e| WeldrsError::Training {
            stage: "em",
            message: format!("Missing count column: {e}"),
        })?;
    let counts: Vec<f64> = count_series
        .u32()
        .map_err(|e| WeldrsError::Training {
            stage: "em",
            message: format!("Count column type error: {e}"),
        })?
        .into_no_null_iter()
        .map(|v| v as f64)
        .collect();

    let mut current_lambda = lambda;
    let mut results = Vec::new();

    for iteration in 0..training.max_iterations {
        // Pre-compute log Bayes factor lookup tables for numerically stable E-step.
        let mut log_bf_tables = build_log_bf_tables(&comparisons);
        // Neutralize block-excluded comparisons (BF = 1 → log-BF = 0) so they
        // don't bias the posterior in the E-step.
        for (table, &ex) in log_bf_tables.iter_mut().zip(excluded.iter()) {
            if ex {
                table.iter_mut().for_each(|v| *v = 0.0);
            }
        }

        // Pre-compute null-level lookup tables for the M-step.
        let null_tables = build_null_tables(&comparisons);

        // E-step: compute match probability for each agreement pattern (log-domain).
        let match_probs = e_step(&gamma_columns, &log_bf_tables, &comparisons, current_lambda)?;

        // M-step: update parameters in place (no Comparison cloning).
        let (new_lambda, max_change, _total_match) = m_step(
            &gamma_columns,
            &counts,
            &match_probs,
            &mut comparisons,
            &null_tables,
        )?;

        // Hold lambda fixed if requested; otherwise adopt the M-step estimate.
        if !opts.fix_probability_two_random_records_match {
            current_lambda = new_lambda;
        }

        log::debug!(
            "EM iteration {iteration}: lambda={current_lambda:.6}, max_change={max_change:.6}"
        );

        if training.store_history || max_change < training.em_convergence {
            results.push(EmIterationResult {
                iteration,
                lambda: current_lambda,
                max_change,
                comparisons: comparisons.clone(),
            });
        }

        if max_change < training.em_convergence {
            break;
        }
    }

    // Ensure we always have a final state — even if `max_iterations == 0`, or
    // `store_history` is off and we never pushed during the loop.
    if results.is_empty() {
        results.push(EmIterationResult {
            iteration: 0,
            lambda: current_lambda,
            max_change: f64::NAN,
            comparisons,
        });
    }
    let final_result = results
        .last()
        .cloned()
        .expect("results is non-empty after the guard above");
    let history = if training.store_history {
        Some(results)
    } else {
        None
    };
    Ok(EmOutcome {
        final_result,
        history,
    })
}

/// Build a lookup table for each comparison: `bf_tables[comp_idx][gamma_val + 1] = bayes_factor`.
///
/// This converts the O(L) linear scan in `level_bayes_factor` to O(1).
/// Also used by `predict_direct()` for direct BF computation.
pub fn build_bf_tables(comparisons: &[Comparison]) -> Vec<Vec<f64>> {
    comparisons
        .iter()
        .map(|comp| {
            let max_cv = comp
                .comparison_levels
                .iter()
                .map(|l| l.comparison_vector_value)
                .max()
                .unwrap_or(0);
            // Index space: gamma_val + 1 (since null = -1 → index 0).
            let size = (max_cv + 2) as usize;
            let mut table = vec![1.0f64; size];
            for level in &comp.comparison_levels {
                let idx = (level.comparison_vector_value + 1) as usize;
                if idx < size {
                    table[idx] = if level.is_null_level {
                        1.0
                    } else {
                        level.bayes_factor().unwrap_or(1.0)
                    };
                }
            }
            table
        })
        .collect()
}

/// Build a log2-domain Bayes factor lookup table.
///
/// `log2_bf_tables[comp_idx][gamma_val + 1] = log2(bayes_factor)`.
/// Used by `predict_direct()` to replace per-row `bf.log2()` calls with
/// O(1) table lookups.
pub fn build_log2_bf_tables(comparisons: &[Comparison]) -> Vec<Vec<f64>> {
    comparisons
        .iter()
        .map(|comp| {
            let max_cv = comp
                .comparison_levels
                .iter()
                .map(|l| l.comparison_vector_value)
                .max()
                .unwrap_or(0);
            let size = (max_cv + 2) as usize;
            let mut table = vec![0.0f64; size]; // log2(1.0) = 0.0
            for level in &comp.comparison_levels {
                let idx = (level.comparison_vector_value + 1) as usize;
                if idx < size {
                    table[idx] = if level.is_null_level {
                        0.0 // log2(1.0) — neutral
                    } else {
                        level.bayes_factor().unwrap_or(1.0).log2()
                    };
                }
            }
            table
        })
        .collect()
}

/// Build a log-domain Bayes factor lookup table for the E-step.
///
/// `log_bf_tables[comp_idx][gamma_val + 1] = ln(bayes_factor)`.
/// Using log-domain prevents numerical overflow/underflow when many
/// comparisons are multiplied together.
fn build_log_bf_tables(comparisons: &[Comparison]) -> Vec<Vec<f64>> {
    comparisons
        .iter()
        .map(|comp| {
            let max_cv = comp
                .comparison_levels
                .iter()
                .map(|l| l.comparison_vector_value)
                .max()
                .unwrap_or(0);
            let size = (max_cv + 2) as usize;
            let mut table = vec![0.0f64; size]; // ln(1.0) = 0.0
            for level in &comp.comparison_levels {
                let idx = (level.comparison_vector_value + 1) as usize;
                if idx < size {
                    table[idx] = if level.is_null_level {
                        0.0 // ln(1.0) — neutral
                    } else {
                        level.bayes_factor().unwrap_or(1.0).ln()
                    };
                }
            }
            table
        })
        .collect()
}

/// Build a lookup table indicating which gamma values correspond to null levels.
/// `null_tables[comp_idx][gamma_val + 1] = is_null`.
fn build_null_tables(comparisons: &[Comparison]) -> Vec<Vec<bool>> {
    comparisons
        .iter()
        .map(|comp| {
            let max_cv = comp
                .comparison_levels
                .iter()
                .map(|l| l.comparison_vector_value)
                .max()
                .unwrap_or(0);
            let size = (max_cv + 2) as usize;
            let mut table = vec![false; size];
            for level in &comp.comparison_levels {
                let idx = (level.comparison_vector_value + 1) as usize;
                if idx < size && level.is_null_level {
                    table[idx] = true;
                }
            }
            table
        })
        .collect()
}

/// E-step: compute the match probability for each agreement pattern.
///
/// Uses log-domain computation (`ln(BF)` sums instead of BF products) for
/// numerical stability, preventing silent overflow/underflow with many
/// comparisons. Converts to probability via a numerically stable sigmoid.
///
/// Returns a Vec<f64> aligned with the rows of `pattern_counts`.
fn e_step(
    gamma_columns: &[Vec<i8>],
    log_bf_tables: &[Vec<f64>],
    comparisons: &[Comparison],
    lambda: f64,
) -> Result<Vec<f64>> {
    let n_rows = gamma_columns.first().map_or(0, |c| c.len());
    let log_prior_odds = probability::prob_to_bayes_factor(lambda).ln();
    let n_comps = comparisons.len();

    let match_probs: Vec<f64> = (0..n_rows)
        .into_par_iter()
        .map(|row| {
            let mut log_odds = log_prior_odds;
            for comp_idx in 0..n_comps {
                let gamma_val = gamma_columns[comp_idx][row];
                let idx = (gamma_val + 1) as usize;
                let table = &log_bf_tables[comp_idx];
                let log_bf = if idx < table.len() { table[idx] } else { 0.0 };
                log_odds += log_bf;
            }
            // Numerically stable sigmoid: avoids exp(large) overflow.
            if log_odds >= 0.0 {
                1.0 / (1.0 + (-log_odds).exp())
            } else {
                let e = log_odds.exp();
                e / (1.0 + e)
            }
        })
        .collect();

    Ok(match_probs)
}

/// Lightweight result from parallel M-step computation for a single comparison.
/// Avoids cloning the full `Comparison` struct during parallel iteration.
struct MStepCompUpdate {
    /// `(new_m, new_u)` for each comparison level. `None` if unchanged
    /// (null level or fixed parameter).
    level_updates: Vec<(Option<f64>, Option<f64>)>,
    max_change: f64,
}

/// Minimum probability for m/u parameters to prevent Bayes factor singularities
/// (BF=0 or BF=infinity) that cause NaN in log-domain computations.
/// Matches Splink's `LEVEL_PROB_CLIP` approach.
const PROB_CLAMP_MIN: f64 = 1e-6;
const PROB_CLAMP_MAX: f64 = 1.0 - 1e-6;

/// M-step: update m, u, and lambda parameters from the E-step match probabilities.
///
/// Uses a single-pass accumulation per comparison with per-level accumulators
/// indexed by gamma value. Updates comparisons in place instead of cloning,
/// eliminating String allocation overhead from the parallel phase.
///
/// Returns (updated lambda, max parameter change, weighted match total).
fn m_step(
    gamma_columns: &[Vec<i8>],
    counts: &[f64],
    match_probs: &[f64],
    comparisons: &mut [Comparison],
    null_tables: &[Vec<bool>],
) -> Result<(f64, f64, f64)> {
    // Compute total_match and total_count once, shared with the lambda update
    // below. This avoids a redundant O(n_patterns) pass after the parallel
    // per-comparison section.
    let mut total_match = 0.0_f64;
    let mut total_count = 0.0_f64;
    for (mp, c) in match_probs.iter().zip(counts.iter()) {
        total_match += mp * c;
        total_count += c;
    }

    // Parallel: compute updates without cloning comparisons.
    let updates: Vec<MStepCompUpdate> = comparisons
        .par_iter()
        .enumerate()
        .map(|(comp_idx, comp)| {
            let gammas = &gamma_columns[comp_idx];
            let null_table = &null_tables[comp_idx];
            let mut local_max_change = 0.0_f64;

            let max_cv = comp
                .comparison_levels
                .iter()
                .map(|l| l.comparison_vector_value)
                .max()
                .unwrap_or(0);
            let table_size = (max_cv + 2) as usize;

            // Single-pass: accumulate per-level match/non-match weighted counts.
            let mut level_match = vec![0.0f64; table_size];
            let mut level_non_match = vec![0.0f64; table_size];
            let mut total_match_weight = 0.0_f64;
            let mut total_non_match_weight = 0.0_f64;

            for (row, &mp) in match_probs.iter().enumerate() {
                let gv = gammas[row];
                let idx = (gv + 1) as usize;
                let is_null = idx < null_table.len() && null_table[idx];
                if is_null {
                    continue;
                }
                let weighted_match = mp * counts[row];
                let weighted_non_match = (1.0 - mp) * counts[row];
                total_match_weight += weighted_match;
                total_non_match_weight += weighted_non_match;
                if idx < table_size {
                    level_match[idx] += weighted_match;
                    level_non_match[idx] += weighted_non_match;
                }
            }

            let mut level_updates = Vec::with_capacity(comp.comparison_levels.len());
            for level in &comp.comparison_levels {
                if level.is_null_level {
                    level_updates.push((None, None));
                    continue;
                }

                let idx = (level.comparison_vector_value + 1) as usize;
                let lm = if idx < table_size {
                    level_match[idx]
                } else {
                    0.0
                };
                let lnm = if idx < table_size {
                    level_non_match[idx]
                } else {
                    0.0
                };

                let new_m = if !level.fix_m_probability {
                    let m = if total_match_weight > 0.0 {
                        (lm / total_match_weight).clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
                    } else {
                        level.m_probability.unwrap_or(0.0)
                    };
                    if let Some(old_m) = level.m_probability {
                        local_max_change = local_max_change.max((m - old_m).abs());
                    }
                    Some(m)
                } else {
                    None
                };

                let new_u = if !level.fix_u_probability {
                    let u = if total_non_match_weight > 0.0 {
                        (lnm / total_non_match_weight).clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
                    } else {
                        level.u_probability.unwrap_or(0.0)
                    };
                    if let Some(old_u) = level.u_probability {
                        local_max_change = local_max_change.max((u - old_u).abs());
                    }
                    Some(u)
                } else {
                    None
                };

                level_updates.push((new_m, new_u));
            }

            MStepCompUpdate {
                level_updates,
                max_change: local_max_change,
            }
        })
        .collect();

    // Sequential: apply updates to comparisons in place.
    let mut max_change = 0.0_f64;
    for (comp_idx, update) in updates.into_iter().enumerate() {
        max_change = max_change.max(update.max_change);
        for (level_idx, (new_m, new_u)) in update.level_updates.into_iter().enumerate() {
            let level = &mut comparisons[comp_idx].comparison_levels[level_idx];
            if let Some(m) = new_m {
                level.m_probability = Some(m);
            }
            if let Some(u) = new_u {
                level.u_probability = Some(u);
            }
        }
    }

    // Update lambda using pre-computed totals.
    let new_lambda = if total_count > 0.0 {
        total_match / total_count
    } else {
        lambda_from_comparisons(comparisons)
    };

    Ok((new_lambda, max_change, total_match))
}

fn lambda_from_comparisons(_comparisons: &[Comparison]) -> f64 {
    0.0001 // fallback
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison::ComparisonBuilder;
    use crate::settings::TrainingSettings;

    /// Build hand-crafted agreement pattern counts for 2 comparisons (first_name, last_name),
    /// each with 2 non-null levels (exact=1, else=0).
    fn make_pattern_counts() -> LazyFrame {
        df!(
            "gamma_first_name" => [1i8, 1, 0, 0],
            "gamma_last_name" => [1i8, 0, 1, 0],
            "__count" => [50u32, 30, 20, 900],
        )
        .unwrap()
        .lazy()
    }

    /// Options for unit tests that intentionally exercise both m and u
    /// movement (the free-function default would fix u, per Splink).
    fn u_moves() -> EmRunOptions {
        EmRunOptions {
            fix_u_probabilities: false,
            ..Default::default()
        }
    }

    fn make_comparisons() -> Vec<Comparison> {
        vec![
            ComparisonBuilder::new("first_name")
                .null_level()
                .exact_match_level()
                .else_level()
                .build()
                .unwrap(),
            ComparisonBuilder::new("last_name")
                .null_level()
                .exact_match_level()
                .else_level()
                .build()
                .unwrap(),
        ]
    }

    #[test]
    fn test_em_converges() {
        let cv = make_pattern_counts();
        let comparisons = make_comparisons();
        let training = TrainingSettings {
            em_convergence: 0.001,
            max_iterations: 100,
            ..Default::default()
        };

        let results =
            expectation_maximization(&cv, comparisons, 0.05, &training, "gamma_", &[], &u_moves())
                .unwrap();

        let last = &results.final_result;
        assert!(
            last.max_change < training.em_convergence,
            "EM should converge, max_change={}, threshold={}",
            last.max_change,
            training.em_convergence
        );
    }

    #[test]
    fn test_em_m_increases_for_match_level() {
        let cv = make_pattern_counts();
        let comparisons = make_comparisons();
        let training = TrainingSettings {
            em_convergence: 0.0001,
            max_iterations: 25,
            ..Default::default()
        };

        // Record initial m for exact match level of first_name
        let initial_m = comparisons[0]
            .comparison_levels
            .iter()
            .find(|l| l.comparison_vector_value == 1)
            .unwrap()
            .m_probability
            .unwrap();

        let results =
            expectation_maximization(&cv, comparisons, 0.05, &training, "gamma_", &[], &u_moves())
                .unwrap();

        let final_m = results.final_result.comparisons[0]
            .comparison_levels
            .iter()
            .find(|l| l.comparison_vector_value == 1)
            .unwrap()
            .m_probability
            .unwrap();

        // With the agreement pattern data, m for the exact level should be high
        // (most matches agree on first_name)
        assert!(
            final_m > initial_m * 0.5,
            "m for exact match should remain substantial, initial={initial_m}, final={final_m}"
        );
    }

    #[test]
    fn test_em_u_less_than_m_for_match_level() {
        let cv = make_pattern_counts();
        let comparisons = make_comparisons();
        let training = TrainingSettings {
            em_convergence: 0.0001,
            max_iterations: 100,
            ..Default::default()
        };

        let results =
            expectation_maximization(&cv, comparisons, 0.05, &training, "gamma_", &[], &u_moves())
                .unwrap();

        let final_comp = &results.final_result.comparisons[0];
        let exact_level = final_comp
            .comparison_levels
            .iter()
            .find(|l| l.comparison_vector_value == 1)
            .unwrap();

        let final_m = exact_level.m_probability.unwrap();
        let final_u = exact_level.u_probability.unwrap();

        // For the exact match level: u should be much less than m
        // (matches agree much more often than non-matches)
        assert!(
            final_u < final_m,
            "u should be less than m for exact match level, u={final_u}, m={final_m}"
        );
    }

    #[test]
    fn test_em_columns_to_fix() {
        let cv = make_pattern_counts();
        let comparisons = make_comparisons();
        let training = TrainingSettings {
            em_convergence: 0.0001,
            max_iterations: 25,
            ..Default::default()
        };

        // Fix first_name (it overlaps with the training blocking rule)
        let initial_fn_levels: Vec<(Option<f64>, Option<f64>)> = comparisons[0]
            .comparison_levels
            .iter()
            .map(|l| (l.m_probability, l.u_probability))
            .collect();

        // Default opts (Splink): fix_u_probabilities=true. first_name overlaps
        // the block (m fixed) and u is globally fixed, so both stay put.
        let results = expectation_maximization(
            &cv,
            comparisons,
            0.05,
            &training,
            "gamma_",
            &["first_name".to_string()],
            &EmRunOptions::default(),
        )
        .unwrap();

        let final_comps = &results.final_result.comparisons;

        // first_name comparison should be unchanged (fixed)
        for (i, level) in final_comps[0].comparison_levels.iter().enumerate() {
            assert_eq!(
                level.m_probability, initial_fn_levels[i].0,
                "Fixed comparison m should not change"
            );
            assert_eq!(
                level.u_probability, initial_fn_levels[i].1,
                "Fixed comparison u should not change"
            );
        }

        // last_name comparison should have changed
        let last_name_changed = final_comps[1]
            .comparison_levels
            .iter()
            .any(|l| !l.is_null_level && !l.fix_m_probability);
        assert!(
            last_name_changed,
            "Last name comparison should not be fixed"
        );
    }

    #[test]
    fn test_em_overlap_fixes_only_m_when_u_unfixed() {
        // With fix_u_probabilities=false, a block-overlapping comparison should
        // have ONLY its m held fixed — its u is free to move (the §2.2 fix).
        let cv = make_pattern_counts();
        let comparisons = make_comparisons();
        let training = TrainingSettings {
            em_convergence: 0.0001,
            max_iterations: 25,
            ..Default::default()
        };

        let initial_fn: Vec<(Option<f64>, Option<f64>)> = comparisons[0]
            .comparison_levels
            .iter()
            .map(|l| (l.m_probability, l.u_probability))
            .collect();

        let results = expectation_maximization(
            &cv,
            comparisons,
            0.05,
            &training,
            "gamma_",
            &["first_name".to_string()],
            &u_moves(),
        )
        .unwrap();

        let final_fn = &results.final_result.comparisons[0];

        // m unchanged (fixed via overlap)…
        for (i, level) in final_fn.comparison_levels.iter().enumerate() {
            assert_eq!(
                level.m_probability, initial_fn[i].0,
                "Overlapping comparison m must stay fixed"
            );
        }
        // …but u moved for at least one non-null level.
        let u_changed = final_fn
            .comparison_levels
            .iter()
            .enumerate()
            .any(|(i, l)| !l.is_null_level && l.u_probability != initial_fn[i].1);
        assert!(u_changed, "Overlapping comparison u should be free to move");
    }

    #[test]
    fn test_em_deterministic() {
        let training = TrainingSettings {
            em_convergence: 0.0001,
            max_iterations: 50,
            ..Default::default()
        };

        let run = || {
            let cv = make_pattern_counts();
            let comparisons = make_comparisons();
            expectation_maximization(&cv, comparisons, 0.05, &training, "gamma_", &[], &u_moves())
                .unwrap()
        };

        let results_a = run();
        let results_b = run();

        assert_eq!(
            results_a.history.as_ref().map(|h| h.len()),
            results_b.history.as_ref().map(|h| h.len()),
            "Should converge in same number of iterations"
        );

        let last_a = &results_a.final_result;
        let last_b = &results_b.final_result;

        // Use epsilon tolerance for floating-point associativity differences
        // from parallel reduction (~1e-15 per operation).
        let eps = 1e-12;
        assert!(
            (last_a.lambda - last_b.lambda).abs() < eps,
            "Lambda should be near-identical: {} vs {}",
            last_a.lambda,
            last_b.lambda
        );
        assert!(
            (last_a.max_change - last_b.max_change).abs() < eps,
            "Max change should be near-identical: {} vs {}",
            last_a.max_change,
            last_b.max_change
        );

        for (ca, cb) in last_a.comparisons.iter().zip(last_b.comparisons.iter()) {
            for (la, lb) in ca.comparison_levels.iter().zip(cb.comparison_levels.iter()) {
                if let (Some(ma), Some(mb)) = (la.m_probability, lb.m_probability) {
                    assert!(
                        (ma - mb).abs() < eps,
                        "m should be near-identical for {}: {} vs {}",
                        la.label,
                        ma,
                        mb
                    );
                }
                if let (Some(ua), Some(ub)) = (la.u_probability, lb.u_probability) {
                    assert!(
                        (ua - ub).abs() < eps,
                        "u should be near-identical for {}: {} vs {}",
                        la.label,
                        ua,
                        ub
                    );
                }
            }
        }
    }

    #[test]
    fn test_em_lambda_updates() {
        let cv = make_pattern_counts();
        let comparisons = make_comparisons();
        let training = TrainingSettings {
            em_convergence: 0.0001,
            max_iterations: 25,
            ..Default::default()
        };
        let initial_lambda = 0.05;

        let results = expectation_maximization(
            &cv,
            comparisons,
            initial_lambda,
            &training,
            "gamma_",
            &[],
            &u_moves(),
        )
        .unwrap();

        let final_lambda = results.final_result.lambda;
        assert!(
            (final_lambda - initial_lambda).abs() > 1e-6,
            "Lambda should change from initial value, initial={initial_lambda}, final={final_lambda}"
        );
    }

    #[test]
    fn test_em_no_history() {
        let cv = make_pattern_counts();
        let comparisons = make_comparisons();
        let training = TrainingSettings {
            em_convergence: 0.0001,
            max_iterations: 25,
            store_history: false,
        };

        let results =
            expectation_maximization(&cv, comparisons, 0.05, &training, "gamma_", &[], &u_moves())
                .unwrap();

        // With store_history=false, no per-iteration history is retained.
        assert!(
            results.history.is_none(),
            "history should be None when store_history is disabled"
        );

        // The final result should still have valid comparisons.
        assert!(!results.final_result.comparisons.is_empty());
    }
}
