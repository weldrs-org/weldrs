//! U-probability estimation via random sampling.
//!
//! Random record pairs are overwhelmingly non-matches, so the frequency of
//! each comparison level among random pairs is a good estimate of its
//! u-probability.
//!
//! Most users call this via
//! [`Linker::estimate_u_using_random_sampling`](crate::linker::Linker::estimate_u_using_random_sampling)
//! rather than invoking the free function directly.

use polars::prelude::*;

use crate::comparison::Comparison;
use crate::comparison_vectors::compute_comparison_vectors;
use crate::error::{Result, WeldrsError};

/// Estimate u-probabilities by comparing random record pairs.
///
/// Since random pairs are overwhelmingly non-matches, the frequency of each
/// comparison level among random pairs is a good estimate of its u-probability.
///
/// # Errors
///
/// Returns [`WeldrsError::Training`]
/// if the DataFrame has fewer than 2 records.
pub fn estimate_u_using_random_sampling(
    lf: &LazyFrame,
    comparisons: &mut [Comparison],
    max_pairs: usize,
    gamma_prefix: &str,
    unique_id_col: &str,
    seed: Option<u64>,
) -> Result<()> {
    let collected = lf.clone().collect().map_err(|e| WeldrsError::Training {
        stage: "estimate_u",
        message: format!("Failed to collect DataFrame: {e}"),
    })?;
    let n_rows = collected.height();
    if n_rows < 2 {
        return Err(WeldrsError::Training {
            stage: "estimate_u",
            message: "Need at least 2 records to estimate u-values".into(),
        });
    }

    // Determine sample size: we want `max_pairs` pairs, so we sample
    // sqrt(2 * max_pairs) records from each side (cross-join gives n^2 pairs,
    // but we only keep uid_l < uid_r → ~n^2/2 pairs).
    let sample_size = ((2.0 * max_pairs as f64).sqrt().ceil() as usize).min(n_rows);

    let uid_l = format!("{unique_id_col}_l");
    let uid_r = format!("{unique_id_col}_r");

    // Pre-select only uid + comparison columns before sampling and cross-join.
    // This reduces the cross-join payload significantly for wide DataFrames.
    let needed_cols: Vec<PlSmallStr> = {
        let mut cols: Vec<&str> = vec![unique_id_col];
        for comp in comparisons.iter() {
            for ic in &comp.input_columns {
                cols.push(ic.as_str());
            }
        }
        cols.sort_unstable();
        cols.dedup();
        cols.into_iter().map(PlSmallStr::from).collect()
    };

    let slim = collected
        .select(needed_cols)
        .map_err(|e| WeldrsError::Training {
            stage: "estimate_u",
            message: format!("Column selection failed: {e}"),
        })?;

    // Sample and create left/right DataFrames.
    let sampled = slim
        .sample_n_literal(sample_size, false, true, seed)
        .map_err(|e| WeldrsError::Training {
            stage: "estimate_u",
            message: format!("Sampling failed: {e}"),
        })?;

    let left = sampled
        .clone()
        .lazy()
        .select([col("*").name().suffix("_l")]);
    let right = sampled.lazy().select([col("*").name().suffix("_r")]);

    // Cross-join and keep uid_l < uid_r.
    let pairs = left
        .cross_join(right, Some("_cross".into()))
        .filter(col(uid_l.as_str()).lt(col(uid_r.as_str())))
        .limit(max_pairs as u32);

    // Compute comparison vectors.
    let cv = compute_comparison_vectors(pairs, comparisons, gamma_prefix)?;

    // Reduce to agreement-pattern counts, then assign each level's frequency
    // among non-matches to its u-probability. Random pairs are overwhelmingly
    // non-matches, so these frequencies estimate u.
    let (pattern_counts, counts) = crate::training_common::group_agreement_patterns(
        cv,
        comparisons,
        gamma_prefix,
        "estimate_u",
    )?;
    crate::training_common::assign_level_frequencies(
        &pattern_counts,
        &counts,
        comparisons,
        gamma_prefix,
        crate::training_common::ProbKind::U,
        "estimate_u",
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison_level::ComparisonLevel;
    use crate::test_helpers;

    fn test_lf() -> LazyFrame {
        test_helpers::make_test_df().lazy()
    }

    #[test]
    fn test_u_probabilities_updated() {
        let lf = test_lf();
        let mut comparisons = vec![test_helpers::exact_match_comparison("first_name")];

        // Store original u values
        let orig_u: Vec<Option<f64>> = comparisons[0]
            .comparison_levels
            .iter()
            .map(|l| l.u_probability)
            .collect();

        estimate_u_using_random_sampling(
            &lf,
            &mut comparisons,
            100,
            "gamma_",
            "unique_id",
            Some(42),
        )
        .unwrap();

        // Non-null levels should have updated u values (may differ from defaults)
        for level in &comparisons[0].comparison_levels {
            if !level.is_null_level {
                assert!(level.u_probability.is_some());
            }
        }

        // At least one u value should have changed from the default
        let new_u: Vec<Option<f64>> = comparisons[0]
            .comparison_levels
            .iter()
            .map(|l| l.u_probability)
            .collect();
        assert_ne!(orig_u, new_u);
    }

    #[test]
    fn test_u_null_levels_untouched() {
        let lf = test_lf();
        let mut comparisons = vec![test_helpers::exact_match_comparison("first_name")];

        estimate_u_using_random_sampling(
            &lf,
            &mut comparisons,
            100,
            "gamma_",
            "unique_id",
            Some(42),
        )
        .unwrap();

        // Null level (first level) should still have None u
        let null_level = &comparisons[0].comparison_levels[0];
        assert!(null_level.is_null_level);
        assert!(null_level.u_probability.is_none());
    }

    #[test]
    fn test_u_values_reasonable() {
        let lf = test_lf();
        let mut comparisons = vec![test_helpers::exact_match_comparison("first_name")];

        estimate_u_using_random_sampling(
            &lf,
            &mut comparisons,
            100,
            "gamma_",
            "unique_id",
            Some(42),
        )
        .unwrap();

        // Among random pairs, exact matches should be rare (u < 0.5)
        // and the else level should be common (u > 0.5)
        let non_null: Vec<&ComparisonLevel> = comparisons[0]
            .comparison_levels
            .iter()
            .filter(|l| !l.is_null_level)
            .collect();

        let exact_level = non_null
            .iter()
            .find(|l| l.comparison_vector_value == 1)
            .unwrap();
        let else_level = non_null
            .iter()
            .find(|l| l.comparison_vector_value == 0)
            .unwrap();

        assert!(
            exact_level.u_probability.unwrap() < 0.5,
            "Exact match u should be < 0.5, got {}",
            exact_level.u_probability.unwrap()
        );
        assert!(
            else_level.u_probability.unwrap() > 0.5,
            "Else u should be > 0.5, got {}",
            else_level.u_probability.unwrap()
        );
    }
}
