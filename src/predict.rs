//! Fellegi-Sunter scoring.
//!
//! Applies the trained model to comparison vectors, producing a match weight
//! (log2 of the combined Bayes factor) and a match probability for every
//! candidate pair.
//!
//! This is **step 4** of the pipeline — after [`em`](crate::em) training
//! has estimated the model parameters and
//! [`comparison_vectors`](crate::comparison_vectors) has computed gamma
//! columns.
//!
//! Two execution strategies are available via [`PredictMode`]:
//!
//! - [`Lazy`](PredictMode::Lazy) — builds a Polars lazy expression graph
//!   (best for large candidate sets).
//! - [`Direct`](PredictMode::Direct) — eager row-wise scoring via BF
//!   lookup tables (best for small candidate sets where Polars planning
//!   overhead dominates).
//! - [`Auto`](PredictMode::Auto) — picks based on candidate-pair volume
//!   and model size.

use polars::prelude::*;
use rayon::prelude::*;

use crate::comparison::Comparison;
use crate::error::{Result, WeldrsError};
use crate::probability;

/// Execution strategy for prediction scoring.
///
/// `Auto` picks an implementation based on candidate-pair volume and model size.
/// `Lazy` uses Polars expressions.
/// `Direct` uses table lookups and eager row-wise scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PredictMode {
    /// Automatically choose between `Lazy` and `Direct` based on the
    /// number of candidate pairs and comparisons. This is the default.
    #[default]
    Auto,
    /// Use Polars lazy expressions to build a query plan. Best for large
    /// candidate sets (> ~50K pairs) where Polars can optimize the plan.
    Lazy,
    /// Use eager row-wise scoring via precomputed Bayes factor lookup
    /// tables. Best for small candidate sets where Polars planning
    /// overhead would dominate.
    Direct,
}

/// Resolve the effective prediction mode.
///
/// `Auto` uses a conservative crossover tuned for current benchmarks:
/// direct scoring tends to win on smaller candidate sets where Polars
/// expression-planning overhead dominates.
pub fn resolve_predict_mode(
    mode: PredictMode,
    n_pairs: usize,
    n_comparisons: usize,
) -> PredictMode {
    match mode {
        PredictMode::Auto => {
            if n_pairs <= 50_000 && n_comparisons <= 8 {
                PredictMode::Direct
            } else {
                PredictMode::Lazy
            }
        }
        explicit => explicit,
    }
}

fn match_probability_from_log2_odds(log2_odds: f64) -> f64 {
    // Convert log2-odds to natural log-odds, then apply a numerically stable
    // sigmoid. Using exp() instead of powf() is ~3× faster, and the two-branch
    // form avoids overflow for large positive log-odds values.
    let log_odds = log2_odds * std::f64::consts::LN_2;
    if log_odds >= 0.0 {
        1.0 / (1.0 + (-log_odds).exp())
    } else {
        let e = log_odds.exp();
        e / (1.0 + e)
    }
}

fn extract_gamma_columns_i8(
    comparison_vectors: &DataFrame,
    comparisons: &[Comparison],
    gamma_prefix: &str,
) -> Result<Vec<Vec<i8>>> {
    comparisons
        .iter()
        .map(|comp| {
            let col_name = comp.gamma_column_name(gamma_prefix);
            let series =
                comparison_vectors
                    .column(&col_name)
                    .map_err(|e| WeldrsError::Training {
                        stage: "predict",
                        message: format!("Missing gamma column: {e}"),
                    })?;
            let cast = series
                .cast(&DataType::Int8)
                .map_err(|e| WeldrsError::Training {
                    stage: "predict",
                    message: format!("Gamma column cast error: {e}"),
                })?;
            let gammas = cast.i8().map_err(|e| WeldrsError::Training {
                stage: "predict",
                message: format!("Gamma column type error: {e}"),
            })?;
            Ok(gammas.into_iter().map(|v| v.unwrap_or(-1i8)).collect())
        })
        .collect::<Result<Vec<_>>>()
}

/// Score record pairs using the trained Fellegi-Sunter model.
///
/// Adds `match_weight` and `match_probability` columns, plus individual
/// `bf_{name}` columns for each comparison.
///
/// Optionally filters to pairs above a threshold (on match probability or
/// match weight).
///
/// # Errors
///
/// Returns an error if building the Bayes factor expression fails for
/// any comparison.
pub fn predict(
    comparison_vectors: LazyFrame,
    comparisons: &[Comparison],
    lambda: f64,
    gamma_prefix: &str,
    bf_prefix: &str,
    threshold_match_probability: Option<f64>,
    threshold_match_weight: Option<f64>,
) -> Result<LazyFrame> {
    let prior_odds = probability::prob_to_bayes_factor(lambda);
    let log2_prior = prior_odds.log2();

    let mut lf = comparison_vectors;

    // Add individual BF columns for each comparison.
    let mut bf_col_names = Vec::new();
    let bf_exprs: Vec<Expr> = comparisons
        .iter()
        .map(|comp| {
            bf_col_names.push(comp.bf_column_name(bf_prefix));
            comp.bf_expr(gamma_prefix, bf_prefix)
        })
        .collect::<Result<Vec<_>>>()?;
    lf = lf.with_columns(bf_exprs);

    // Compute combined match weight = log2(prior_odds) + sum(log2(BF_i)).
    let mut match_weight_expr = lit(log2_prior);
    for bf_col in &bf_col_names {
        match_weight_expr = match_weight_expr + col(bf_col.as_str()).log(lit(2.0));
    }
    lf = lf.with_column(match_weight_expr.alias("match_weight"));

    // match_probability = 2^match_weight / (1 + 2^match_weight)
    // which is equivalent to bayes_factor_to_prob(2^match_weight)
    let bf_total = lit(2.0_f64).pow(col("match_weight"));
    let match_prob_expr = bf_total.clone() / (lit(1.0_f64) + bf_total);
    lf = lf.with_column(match_prob_expr.alias("match_probability"));

    // Apply thresholds.
    if let Some(thresh) = threshold_match_probability {
        lf = lf.filter(col("match_probability").gt_eq(lit(thresh)));
    }
    if let Some(thresh) = threshold_match_weight {
        lf = lf.filter(col("match_weight").gt_eq(lit(thresh)));
    }

    Ok(lf)
}

/// Score record pairs using direct BF table lookup (bypasses Polars lazy overhead).
///
/// More efficient than the lazy-expression path for datasets under ~100K pairs
/// where Polars query-planning overhead for the `when/then/otherwise` BF
/// expression chains would dominate the actual computation.
///
/// Adds `match_weight`, `match_probability`, and individual `bf_{name}` columns.
/// Optionally filters to pairs above a threshold.
///
/// # Errors
///
/// Returns an error if gamma columns are missing or cannot be cast to `i8`.
pub fn predict_direct(
    comparison_vectors: DataFrame,
    comparisons: &[Comparison],
    lambda: f64,
    gamma_prefix: &str,
    bf_prefix: &str,
    threshold_match_probability: Option<f64>,
    threshold_match_weight: Option<f64>,
) -> Result<DataFrame> {
    let bf_tables = crate::em::build_bf_tables(comparisons);
    let log2_bf_tables = crate::em::build_log2_bf_tables(comparisons);
    let prior_log2 = probability::prob_to_bayes_factor(lambda).log2();
    let n_rows = comparison_vectors.height();
    let n_comps = comparisons.len();

    // Extract gamma columns as i8 arrays (casting from wider integer types if needed).
    let gamma_columns = extract_gamma_columns_i8(&comparison_vectors, comparisons, gamma_prefix)?;

    if threshold_match_probability.is_some() || threshold_match_weight.is_some() {
        // Threshold path: parallel scoring, then sequential filter + collect.
        // Phase 1: score every row in parallel, producing per-row results.
        let row_scores: Vec<(f64, f64, Vec<f64>)> = (0..n_rows)
            .into_par_iter()
            .map(|row| {
                let mut match_weight = prior_log2;
                let mut row_bfs = Vec::with_capacity(n_comps);
                for comp_idx in 0..n_comps {
                    let gv = gamma_columns[comp_idx][row];
                    let idx = (gv + 1) as usize;
                    let bf_table = &bf_tables[comp_idx];
                    let log2_table = &log2_bf_tables[comp_idx];
                    let bf = if idx < bf_table.len() {
                        bf_table[idx]
                    } else {
                        1.0
                    };
                    row_bfs.push(bf);
                    match_weight += if idx < log2_table.len() {
                        log2_table[idx]
                    } else {
                        0.0
                    };
                }
                let match_probability = match_probability_from_log2_odds(match_weight);
                (match_weight, match_probability, row_bfs)
            })
            .collect();

        // Phase 2: sequential filter + collect kept rows.
        let mut kept_row_idx: Vec<IdxSize> = Vec::with_capacity(n_rows);
        let mut bf_values: Vec<Vec<f64>> =
            (0..n_comps).map(|_| Vec::with_capacity(n_rows)).collect();
        let mut match_weights: Vec<f64> = Vec::with_capacity(n_rows);
        let mut match_probs: Vec<f64> = Vec::with_capacity(n_rows);

        for (row, (mw, mp, row_bfs)) in row_scores.into_iter().enumerate() {
            let keep = threshold_match_probability
                .map(|th| mp >= th)
                .unwrap_or(true)
                && threshold_match_weight.map(|th| mw >= th).unwrap_or(true);

            if keep {
                let idx = IdxSize::try_from(row).map_err(|_| WeldrsError::Training {
                    stage: "predict",
                    message: "Too many rows for Polars index type".into(),
                })?;
                kept_row_idx.push(idx);
                match_weights.push(mw);
                match_probs.push(mp);
                for (comp_idx, bf) in row_bfs.into_iter().enumerate() {
                    bf_values[comp_idx].push(bf);
                }
            }
        }

        let idx = IdxCa::from_vec("idx".into(), kept_row_idx);
        let mut df = comparison_vectors
            .take(&idx)
            .map_err(|e| WeldrsError::Training {
                stage: "predict",
                message: format!("Row take failed: {e}"),
            })?;

        for (comp_idx, comp) in comparisons.iter().enumerate() {
            let name = comp.bf_column_name(bf_prefix);
            df.with_column(Column::new(name.into(), &bf_values[comp_idx]))
                .map_err(|e| WeldrsError::Training {
                    stage: "predict",
                    message: format!("Failed to add BF column: {e}"),
                })?;
        }
        df.with_column(Column::new("match_weight".into(), &match_weights))
            .map_err(|e| WeldrsError::Training {
                stage: "predict",
                message: format!("Failed to add match_weight: {e}"),
            })?;
        df.with_column(Column::new("match_probability".into(), &match_probs))
            .map_err(|e| WeldrsError::Training {
                stage: "predict",
                message: format!("Failed to add match_probability: {e}"),
            })?;

        return Ok(df);
    }

    // Fast path: no thresholds. Keep column-wise accumulation; benchmarks show
    // it is faster for dense outputs than row-wise scoring.
    let mut bf_values: Vec<Vec<f64>> = Vec::with_capacity(n_comps);
    let mut match_weights = vec![prior_log2; n_rows];
    for comp_idx in 0..n_comps {
        let bf_table = &bf_tables[comp_idx];
        let log2_table = &log2_bf_tables[comp_idx];
        let gammas = &gamma_columns[comp_idx];
        let mut bfs = Vec::with_capacity(n_rows);
        for (row, &gv) in gammas.iter().enumerate() {
            let idx = (gv + 1) as usize;
            let bf = if idx < bf_table.len() {
                bf_table[idx]
            } else {
                1.0
            };
            bfs.push(bf);
            match_weights[row] += if idx < log2_table.len() {
                log2_table[idx]
            } else {
                0.0
            };
        }
        bf_values.push(bfs);
    }
    let match_probs: Vec<f64> = match_weights
        .iter()
        .map(|&mw| match_probability_from_log2_odds(mw))
        .collect();

    let mut df = comparison_vectors;
    for (comp_idx, comp) in comparisons.iter().enumerate() {
        let name = comp.bf_column_name(bf_prefix);
        df.with_column(Column::new(name.into(), &bf_values[comp_idx]))
            .map_err(|e| WeldrsError::Training {
                stage: "predict",
                message: format!("Failed to add BF column: {e}"),
            })?;
    }
    df.with_column(Column::new("match_weight".into(), &match_weights))
        .map_err(|e| WeldrsError::Training {
            stage: "predict",
            message: format!("Failed to add match_weight: {e}"),
        })?;
    df.with_column(Column::new("match_probability".into(), &match_probs))
        .map_err(|e| WeldrsError::Training {
            stage: "predict",
            message: format!("Failed to add match_probability: {e}"),
        })?;

    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison::ComparisonBuilder;

    fn trained_comparison() -> crate::comparison::Comparison {
        let mut comp = ComparisonBuilder::new("name")
            .null_level()
            .exact_match_level()
            .else_level()
            .build()
            .unwrap();

        // Set known m/u values
        for level in &mut comp.comparison_levels {
            if level.is_null_level {
                continue;
            }
            if level.comparison_vector_value == 1 {
                level.m_probability = Some(0.95);
                level.u_probability = Some(0.01);
            } else {
                level.m_probability = Some(0.05);
                level.u_probability = Some(0.99);
            }
        }
        comp
    }

    fn comparison_vector_lf(gamma_values: &[i32]) -> LazyFrame {
        let uids_l: Vec<i64> = (0..gamma_values.len() as i64).collect();
        let uids_r: Vec<i64> = (100..100 + gamma_values.len() as i64).collect();
        df!(
            "unique_id_l" => &uids_l,
            "unique_id_r" => &uids_r,
            "gamma_name" => gamma_values,
        )
        .unwrap()
        .lazy()
    }

    #[test]
    fn test_predict_adds_columns() {
        let comp = trained_comparison();
        let cv = comparison_vector_lf(&[1, 0]);

        let result = predict(cv, &[comp], 0.0001, "gamma_", "bf_", None, None)
            .unwrap()
            .collect()
            .unwrap();

        let col_names: Vec<&str> = result
            .get_column_names()
            .into_iter()
            .map(|s| s.as_str())
            .collect();
        assert!(col_names.contains(&"match_weight"));
        assert!(col_names.contains(&"match_probability"));
        assert!(col_names.contains(&"bf_name"));
    }

    #[test]
    fn test_predict_correct_values() {
        let comp = trained_comparison();
        let cv = comparison_vector_lf(&[1, 0]); // exact match pair, non-match pair

        // Use lambda=0.1 so the prior doesn't dominate with a single comparison
        let result = predict(cv, &[comp], 0.1, "gamma_", "bf_", None, None)
            .unwrap()
            .collect()
            .unwrap();

        let probs: Vec<Option<f64>> = result
            .column("match_probability")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .collect();

        // Exact match pair should have high probability
        assert!(
            probs[0].unwrap() > 0.5,
            "Exact match pair should have high probability"
        );
        // Non-match pair should have low probability
        assert!(
            probs[1].unwrap() < 0.5,
            "Non-match pair should have low probability"
        );
    }

    #[test]
    fn test_predict_threshold_probability() {
        let comp = trained_comparison();
        let cv = comparison_vector_lf(&[1, 0, 0]);

        let result = predict(cv, &[comp], 0.0001, "gamma_", "bf_", Some(0.5), None)
            .unwrap()
            .collect()
            .unwrap();

        // Only the exact-match pair should survive the threshold
        assert!(result.height() <= 2, "Threshold should filter some pairs");

        // All remaining pairs should have match_probability >= 0.5
        let probs = result.column("match_probability").unwrap().f64().unwrap();
        for p in probs.into_iter().flatten() {
            assert!(p >= 0.5);
        }
    }

    #[test]
    fn test_predict_threshold_weight() {
        let comp = trained_comparison();
        let cv = comparison_vector_lf(&[1, 0, 0]);

        let result = predict(cv, &[comp], 0.0001, "gamma_", "bf_", None, Some(0.0))
            .unwrap()
            .collect()
            .unwrap();

        // All remaining pairs should have match_weight >= 0.0
        let weights = result.column("match_weight").unwrap().f64().unwrap();
        for w in weights.into_iter().flatten() {
            assert!(w >= 0.0);
        }
    }

    #[test]
    #[allow(clippy::cloned_ref_to_slice_refs)]
    fn test_predict_direct_matches_lazy() {
        let comp = trained_comparison();
        let cv_eager = comparison_vector_lf(&[1, 0]).collect().unwrap();

        let lazy_result = predict(
            cv_eager.clone().lazy(),
            &[comp.clone()],
            0.1,
            "gamma_",
            "bf_",
            None,
            None,
        )
        .unwrap()
        .collect()
        .unwrap();

        let direct_result =
            predict_direct(cv_eager.clone(), &[comp], 0.1, "gamma_", "bf_", None, None).unwrap();

        // Both paths should produce the same match probabilities (within f64 tolerance).
        let lazy_probs: Vec<f64> = lazy_result
            .column("match_probability")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        let direct_probs: Vec<f64> = direct_result
            .column("match_probability")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();

        assert_eq!(lazy_probs.len(), direct_probs.len());
        for (lp, dp) in lazy_probs.iter().zip(direct_probs.iter()) {
            assert!(
                (lp - dp).abs() < 1e-10,
                "Lazy vs direct mismatch: {lp} vs {dp}"
            );
        }
    }

    #[test]
    fn test_predict_direct_threshold() {
        let comp = trained_comparison();
        let cv_eager = comparison_vector_lf(&[1, 0, 0]).collect().unwrap();

        let result =
            predict_direct(cv_eager, &[comp], 0.0001, "gamma_", "bf_", Some(0.5), None).unwrap();

        let probs = result.column("match_probability").unwrap().f64().unwrap();
        for p in probs.into_iter().flatten() {
            assert!(p >= 0.5);
        }
    }

    #[test]
    fn test_resolve_predict_mode_auto() {
        assert_eq!(
            resolve_predict_mode(PredictMode::Auto, 10_000, 3),
            PredictMode::Direct
        );
        assert_eq!(
            resolve_predict_mode(PredictMode::Auto, 200_000, 3),
            PredictMode::Lazy
        );
        assert_eq!(
            resolve_predict_mode(PredictMode::Auto, 10_000, 16),
            PredictMode::Lazy
        );
    }
}
