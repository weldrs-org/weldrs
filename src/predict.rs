//! Fellegi-Sunter scoring.
//!
//! Applies the trained model to comparison vectors, producing a match weight
//! (log2 of the combined Bayes factor) and a match probability for every
//! candidate pair.

use polars::prelude::*;

use crate::comparison::Comparison;
use crate::error::Result;
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
}
