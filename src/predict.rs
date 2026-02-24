//! Fellegi-Sunter scoring.
//!
//! Applies the trained model to comparison vectors, producing a match weight
//! (log2 of the combined Bayes factor) and a match probability for every
//! candidate pair.

use polars::prelude::*;

use crate::comparison::Comparison;
use crate::error::{Result, WeldrsError};
use crate::probability;

/// Score record pairs using the trained Fellegi-Sunter model.
///
/// Adds `match_weight` and `match_probability` columns, plus individual
/// `bf_{name}` columns for each comparison.
///
/// Optionally filters to pairs above a threshold (on match probability or
/// match weight).
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
        match_weight_expr = match_weight_expr + col(bf_col.as_str()).log(2.0);
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
pub fn predict_direct(
    comparison_vectors: &DataFrame,
    comparisons: &[Comparison],
    lambda: f64,
    gamma_prefix: &str,
    bf_prefix: &str,
    threshold_match_probability: Option<f64>,
    threshold_match_weight: Option<f64>,
) -> Result<DataFrame> {
    let bf_tables = crate::em::build_bf_tables(comparisons);
    let prior_log2 = probability::prob_to_bayes_factor(lambda).log2();
    let n_rows = comparison_vectors.height();

    // Extract gamma columns as i8 arrays (casting from wider integer types if needed).
    let gamma_columns: Vec<Vec<i8>> = comparisons
        .iter()
        .map(|comp| {
            let col_name = comp.gamma_column_name(gamma_prefix);
            let series = comparison_vectors
                .column(&col_name)
                .map_err(|e| WeldrsError::Training(format!("Missing gamma column: {e}")))?;
            let cast = series
                .cast(&DataType::Int8)
                .map_err(|e| WeldrsError::Training(format!("Gamma column cast error: {e}")))?;
            let gammas = cast
                .i8()
                .map_err(|e| WeldrsError::Training(format!("Gamma column type error: {e}")))?;
            Ok(gammas.into_iter().map(|v| v.unwrap_or(-1i8)).collect())
        })
        .collect::<Result<Vec<_>>>()?;

    // Compute individual BF columns, match_weight, and match_probability.
    let mut bf_vecs: Vec<(String, Vec<f64>)> = Vec::with_capacity(comparisons.len());
    let mut match_weights = vec![prior_log2; n_rows];

    for (comp_idx, comp) in comparisons.iter().enumerate() {
        let bf_col_name = comp.bf_column_name(bf_prefix);
        let table = &bf_tables[comp_idx];
        let gammas = &gamma_columns[comp_idx];
        let mut bfs = Vec::with_capacity(n_rows);

        for (row, &gv) in gammas.iter().enumerate() {
            let idx = (gv + 1) as usize;
            let bf = if idx < table.len() { table[idx] } else { 1.0 };
            bfs.push(bf);
            match_weights[row] += bf.log2();
        }

        bf_vecs.push((bf_col_name, bfs));
    }

    // Compute match probabilities from match weights.
    let match_probs: Vec<f64> = match_weights
        .iter()
        .map(|&mw| {
            let bf = (2.0_f64).powf(mw);
            bf / (1.0 + bf)
        })
        .collect();

    // Build output DataFrame.
    let mut df = comparison_vectors.clone();
    for (name, values) in bf_vecs {
        df.with_column(Column::new(name.into(), &values))
            .map_err(|e| WeldrsError::Training(format!("Failed to add BF column: {e}")))?;
    }
    df.with_column(Column::new("match_weight".into(), &match_weights))
        .map_err(|e| WeldrsError::Training(format!("Failed to add match_weight: {e}")))?;
    df.with_column(Column::new("match_probability".into(), &match_probs))
        .map_err(|e| WeldrsError::Training(format!("Failed to add match_probability: {e}")))?;

    // Apply thresholds using lazy filter for simplicity.
    if let Some(thresh) = threshold_match_probability {
        df = df
            .lazy()
            .filter(col("match_probability").gt_eq(lit(thresh)))
            .collect()
            .map_err(|e| WeldrsError::Training(format!("Threshold filter failed: {e}")))?;
    }
    if let Some(thresh) = threshold_match_weight {
        df = df
            .lazy()
            .filter(col("match_weight").gt_eq(lit(thresh)))
            .collect()
            .map_err(|e| WeldrsError::Training(format!("Threshold filter failed: {e}")))?;
    }

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
            .build();

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

    fn cv_df(gamma_values: &[i32]) -> LazyFrame {
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
        let cv = cv_df(&[1, 0]);

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
        let cv = cv_df(&[1, 0]); // exact match pair, non-match pair

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
        let cv = cv_df(&[1, 0, 0]);

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
        let cv = cv_df(&[1, 0, 0]);

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
    fn test_predict_direct_matches_lazy() {
        let comp = trained_comparison();
        let cv_eager = cv_df(&[1, 0]).collect().unwrap();

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
            predict_direct(&cv_eager, &[comp], 0.1, "gamma_", "bf_", None, None).unwrap();

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
        let cv_eager = cv_df(&[1, 0, 0]).collect().unwrap();

        let result =
            predict_direct(&cv_eager, &[comp], 0.0001, "gamma_", "bf_", Some(0.5), None).unwrap();

        let probs = result.column("match_probability").unwrap().f64().unwrap();
        for p in probs.into_iter().flatten() {
            assert!(p >= 0.5);
        }
    }
}
