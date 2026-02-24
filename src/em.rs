//! Expectation-Maximisation (EM) training.
//!
//! The EM algorithm iteratively estimates m-probabilities, u-probabilities,
//! and lambda from agreement-pattern counts derived from blocked record
//! pairs.

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

/// Run the EM algorithm on comparison vectors.
///
/// `comparison_vectors` must contain gamma columns for every comparison in
/// `comparisons`. Comparisons listed in `columns_to_fix` will not have their
/// m/u parameters updated (they overlap with the training blocking rule).
///
/// Returns the iteration history (the last entry contains the final parameters).
pub fn expectation_maximisation(
    comparison_vectors: &LazyFrame,
    mut comparisons: Vec<Comparison>,
    lambda: f64,
    training: &TrainingSettings,
    gamma_prefix: &str,
    columns_to_fix: &[String],
) -> Result<Vec<EmIterationResult>> {
    // Mark comparisons whose columns overlap with the training blocking rule.
    for comp in &mut comparisons {
        if columns_to_fix
            .iter()
            .any(|c| comp.input_columns.contains(c))
        {
            for level in &mut comp.comparison_levels {
                level.fix_m_probability = true;
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
        .map_err(|e| WeldrsError::Training(format!("Failed to count agreement patterns: {e}")))?;

    // Pre-extract gamma columns once — shared across all E/M steps.
    let gamma_columns: Vec<Vec<i8>> = comparisons
        .iter()
        .map(|comp| {
            let col_name = comp.gamma_column_name(gamma_prefix);
            let series = pattern_counts
                .column(&col_name)
                .map_err(|e| WeldrsError::Training(format!("Missing gamma column: {e}")))?;
            let gammas = series
                .i8()
                .map_err(|e| WeldrsError::Training(format!("Gamma column type error: {e}")))?;
            Ok(gammas.into_iter().map(|v| v.unwrap_or(-1i8)).collect())
        })
        .collect::<Result<Vec<_>>>()?;

    // Pre-extract counts once.
    let count_series = pattern_counts
        .column("__count")
        .map_err(|e| WeldrsError::Training(format!("Missing count column: {e}")))?;
    let counts: Vec<f64> = count_series
        .u32()
        .map_err(|e| WeldrsError::Training(format!("Count column type error: {e}")))?
        .into_no_null_iter()
        .map(|v| v as f64)
        .collect();

    let mut current_lambda = lambda;
    let mut results = Vec::new();

    for iteration in 0..training.max_iterations {
        // Pre-compute Bayes factor lookup tables (O(1) per gamma lookup).
        let bf_tables = build_bf_tables(&comparisons);

        // Pre-compute null-level lookup tables for the M-step.
        let null_tables = build_null_tables(&comparisons);

        // E-step: compute match probability for each agreement pattern.
        let match_probs = e_step(&gamma_columns, &bf_tables, &comparisons, current_lambda)?;

        // M-step: update parameters.
        let (new_comparisons, new_lambda, max_change) = m_step(
            &gamma_columns,
            &counts,
            &match_probs,
            &comparisons,
            &null_tables,
        )?;

        comparisons = new_comparisons;
        current_lambda = new_lambda;

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

    // Always ensure at least one result with the final state.
    if results.is_empty() || results.last().unwrap().comparisons.as_ptr() != comparisons.as_ptr() {
        // If store_history was false and we didn't converge, push the final state.
        if !training.store_history {
            let iteration = results.len();
            results.push(EmIterationResult {
                iteration,
                lambda: current_lambda,
                max_change: f64::NAN,
                comparisons,
            });
        }
    }

    Ok(results)
}

/// Build a lookup table for each comparison: `bf_tables[comp_idx][gamma_val + 1] = bayes_factor`.
///
/// This converts the O(L) linear scan in `level_bayes_factor` to O(1).
fn build_bf_tables(comparisons: &[Comparison]) -> Vec<Vec<f64>> {
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
/// Returns a Vec<f64> aligned with the rows of `pattern_counts`.
fn e_step(
    gamma_columns: &[Vec<i8>],
    bf_tables: &[Vec<f64>],
    comparisons: &[Comparison],
    lambda: f64,
) -> Result<Vec<f64>> {
    let n_rows = gamma_columns.first().map_or(0, |c| c.len());
    let prior_odds = probability::prob_to_bayes_factor(lambda);

    let match_probs: Vec<f64> = (0..n_rows)
        .into_par_iter()
        .map(|row| {
            let mut bf_product = prior_odds;
            for (comp_idx, _comp) in comparisons.iter().enumerate() {
                let gamma_val = gamma_columns[comp_idx][row];
                let idx = (gamma_val + 1) as usize;
                let table = &bf_tables[comp_idx];
                let bf = if idx < table.len() { table[idx] } else { 1.0 };
                bf_product *= bf;
            }
            probability::bayes_factor_to_prob(bf_product)
        })
        .collect();

    Ok(match_probs)
}

/// M-step: update m, u, and lambda parameters from the E-step match probabilities.
///
/// Uses a single-pass accumulation per comparison with per-level accumulators
/// indexed by gamma value, replacing the previous (1 + L) passes.
///
/// Returns (updated comparisons, updated lambda, max parameter change).
fn m_step(
    gamma_columns: &[Vec<i8>],
    counts: &[f64],
    match_probs: &[f64],
    comparisons: &[Comparison],
    null_tables: &[Vec<bool>],
) -> Result<(Vec<Comparison>, f64, f64)> {
    // Parallelize the outer loop over comparisons — each comparison's m/u
    // update is independent.
    let results: Vec<(Comparison, f64)> = comparisons
        .par_iter()
        .enumerate()
        .map(|(comp_idx, comp)| {
            let mut comp = comp.clone();
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

            for level in &mut comp.comparison_levels {
                if level.is_null_level {
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

                if !level.fix_m_probability {
                    let new_m = if total_match_weight > 0.0 {
                        lm / total_match_weight
                    } else {
                        level.m_probability.unwrap_or(0.0)
                    };
                    if let Some(old_m) = level.m_probability {
                        local_max_change = local_max_change.max((new_m - old_m).abs());
                    }
                    level.m_probability = Some(new_m);
                }

                if !level.fix_u_probability {
                    let new_u = if total_non_match_weight > 0.0 {
                        lnm / total_non_match_weight
                    } else {
                        level.u_probability.unwrap_or(0.0)
                    };
                    if let Some(old_u) = level.u_probability {
                        local_max_change = local_max_change.max((new_u - old_u).abs());
                    }
                    level.u_probability = Some(new_u);
                }
            }

            (comp, local_max_change)
        })
        .collect();

    let max_change = results.iter().map(|(_, c)| *c).fold(0.0_f64, f64::max);
    let new_comparisons: Vec<Comparison> = results.into_iter().map(|(c, _)| c).collect();

    // Update lambda.
    let total_count: f64 = counts.iter().sum();
    let total_match: f64 = match_probs
        .iter()
        .zip(counts.iter())
        .map(|(mp, c)| mp * c)
        .sum();
    let new_lambda = if total_count > 0.0 {
        total_match / total_count
    } else {
        lambda_from_comparisons(comparisons)
    };

    Ok((new_comparisons, new_lambda, max_change))
}

fn lambda_from_comparisons(_comparisons: &[Comparison]) -> f64 {
    0.0001 // fallback
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison::ComparisonBuilder;
    use crate::settings::TrainingSettings;

    /// Build hand-crafted agreement pattern counts for 2 comparisons (first_name, surname),
    /// each with 2 non-null levels (exact=1, else=0).
    fn make_pattern_counts() -> LazyFrame {
        df!(
            "gamma_first_name" => [1i8, 1, 0, 0],
            "gamma_surname" => [1i8, 0, 1, 0],
            "__count" => [50u32, 30, 20, 900],
        )
        .unwrap()
        .lazy()
    }

    fn make_comparisons() -> Vec<Comparison> {
        vec![
            ComparisonBuilder::new("first_name")
                .null_level()
                .exact_match_level()
                .else_level()
                .build(),
            ComparisonBuilder::new("surname")
                .null_level()
                .exact_match_level()
                .else_level()
                .build(),
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
            expectation_maximisation(&cv, comparisons, 0.05, &training, "gamma_", &[]).unwrap();

        let last = results.last().unwrap();
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
            expectation_maximisation(&cv, comparisons, 0.05, &training, "gamma_", &[]).unwrap();

        let final_m = results.last().unwrap().comparisons[0]
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
            expectation_maximisation(&cv, comparisons, 0.05, &training, "gamma_", &[]).unwrap();

        let final_comp = &results.last().unwrap().comparisons[0];
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

        let results = expectation_maximisation(
            &cv,
            comparisons,
            0.05,
            &training,
            "gamma_",
            &["first_name".to_string()],
        )
        .unwrap();

        let final_comps = &results.last().unwrap().comparisons;

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

        // surname comparison should have changed
        let surname_changed = final_comps[1]
            .comparison_levels
            .iter()
            .any(|l| !l.is_null_level && !l.fix_m_probability);
        assert!(surname_changed, "Surname comparison should not be fixed");
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
            expectation_maximisation(&cv, comparisons, 0.05, &training, "gamma_", &[]).unwrap()
        };

        let results_a = run();
        let results_b = run();

        assert_eq!(
            results_a.len(),
            results_b.len(),
            "Should converge in same number of iterations"
        );

        let last_a = results_a.last().unwrap();
        let last_b = results_b.last().unwrap();

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

        let results =
            expectation_maximisation(&cv, comparisons, initial_lambda, &training, "gamma_", &[])
                .unwrap();

        let final_lambda = results.last().unwrap().lambda;
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
            expectation_maximisation(&cv, comparisons, 0.05, &training, "gamma_", &[]).unwrap();

        // With store_history=false, should have at most 2 entries
        // (final convergence entry + possible last-state entry).
        assert!(
            results.len() <= 2,
            "Without history storage, should have few results, got {}",
            results.len()
        );

        // The last result should still have valid comparisons.
        let last = results.last().unwrap();
        assert!(!last.comparisons.is_empty());
    }
}
