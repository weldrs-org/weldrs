//! Gamma (comparison vector) computation.
//!
//! For each blocked record pair, this module evaluates every
//! [`Comparison`] and appends a gamma column
//! whose integer value indicates which comparison level matched.

use polars::prelude::*;

use crate::comparison::Comparison;
use crate::error::Result;

/// Compute gamma (comparison vector) columns for blocked pairs.
///
/// For each [`Comparison`], appends a `gamma_{name}` column to the DataFrame,
/// where each value indicates which comparison level matched.
pub fn compute_comparison_vectors(
    blocked_pairs: LazyFrame,
    comparisons: &[Comparison],
    gamma_prefix: &str,
) -> Result<LazyFrame> {
    let gamma_exprs: Vec<Expr> = comparisons
        .iter()
        .map(|c| c.gamma_expr(gamma_prefix))
        .collect::<Result<Vec<_>>>()?;
    Ok(blocked_pairs.with_columns(gamma_exprs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers;

    #[test]
    fn test_gamma_columns_added() {
        let comp1 = test_helpers::exact_match_comparison("first_name");
        let comp2 = test_helpers::exact_match_comparison("surname");

        let df = df!(
            "unique_id_l" => [1i64],
            "unique_id_r" => [2i64],
            "first_name_l" => ["Alice"],
            "first_name_r" => ["Bob"],
            "surname_l" => ["Smith"],
            "surname_r" => ["Smith"],
        )
        .unwrap()
        .lazy();

        let result = compute_comparison_vectors(df, &[comp1, comp2], "gamma_")
            .unwrap()
            .collect()
            .unwrap();

        let col_names: Vec<&str> = result
            .get_column_names()
            .into_iter()
            .map(|s| s.as_str())
            .collect();
        assert!(col_names.contains(&"gamma_first_name"));
        assert!(col_names.contains(&"gamma_surname"));
    }

    #[test]
    fn test_comparison_vectors_multiple_comparisons() {
        let comp1 = test_helpers::exact_match_comparison("first_name");
        let comp2 = test_helpers::exact_match_comparison("surname");
        let comp3 = test_helpers::exact_match_comparison("city");

        let df = df!(
            "unique_id_l" => [1i64, 2, 3],
            "unique_id_r" => [4i64, 5, 6],
            "first_name_l" => ["Alice", "Bob", "Carol"],
            "first_name_r" => ["Alice", "Charlie", "Carol"],
            "surname_l" => ["Smith", "Jones", "Brown"],
            "surname_r" => ["Smith", "Jones", "White"],
            "city_l" => ["London", "Paris", "Berlin"],
            "city_r" => ["London", "Rome", "Berlin"],
        )
        .unwrap()
        .lazy();

        let result = compute_comparison_vectors(df, &[comp1, comp2, comp3], "gamma_")
            .unwrap()
            .collect()
            .unwrap();

        // All 3 gamma columns present.
        let col_names: Vec<&str> = result
            .get_column_names()
            .into_iter()
            .map(|s| s.as_str())
            .collect();
        assert!(col_names.contains(&"gamma_first_name"));
        assert!(col_names.contains(&"gamma_surname"));
        assert!(col_names.contains(&"gamma_city"));

        let get_gammas = |name: &str| -> Vec<Option<i8>> {
            result
                .column(name)
                .unwrap()
                .i8()
                .unwrap()
                .into_iter()
                .collect()
        };

        // Alice==Alice(1), Bob!=Charlie(0), Carol==Carol(1)
        assert_eq!(
            get_gammas("gamma_first_name"),
            vec![Some(1i8), Some(0i8), Some(1i8)]
        );
        // Smith==Smith(1), Jones==Jones(1), Brown!=White(0)
        assert_eq!(get_gammas("gamma_surname"), vec![Some(1i8), Some(1i8), Some(0i8)]);
        // London==London(1), Paris!=Rome(0), Berlin==Berlin(1)
        assert_eq!(get_gammas("gamma_city"), vec![Some(1i8), Some(0i8), Some(1i8)]);
    }

    #[test]
    fn test_gamma_values_correct() {
        let comp = test_helpers::exact_match_comparison("name");

        let df = df!(
            "unique_id_l" => [1i64, 2],
            "unique_id_r" => [3i64, 4],
            "name_l" => ["Alice", "Bob"],
            "name_r" => ["Alice", "Carol"],
        )
        .unwrap()
        .lazy();

        let result = compute_comparison_vectors(df, &[comp], "gamma_")
            .unwrap()
            .collect()
            .unwrap();

        let gammas: Vec<Option<i8>> = result
            .column("gamma_name")
            .unwrap()
            .i8()
            .unwrap()
            .into_iter()
            .collect();

        // Alice==Alice → exact(1), Bob!=Carol → else(0)
        assert_eq!(gammas, vec![Some(1i8), Some(0i8)]);
    }
}
