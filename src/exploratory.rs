//! Exploratory data profiling.
//!
//! Lightweight, backend-agnostic helpers for understanding input data before
//! configuring a model: per-column completeness and cardinality, and value
//! frequency distributions. These return plain Polars `DataFrame`s (no charts),
//! mirroring the intent of Splink's `profile_columns` / `completeness_chart`.

use polars::prelude::*;

use crate::error::{Result, WeldrsError};

/// Profile the given columns: row count, non-null count, distinct count, and
/// null / distinct fractions.
///
/// Returns a DataFrame with one row per column and columns
/// `[column, n_rows, n_non_null, n_distinct, null_fraction, distinct_fraction]`.
///
/// # Errors
///
/// Returns an error if a column is missing or a Polars op fails.
pub fn profile_columns(lf: &LazyFrame, columns: &[&str]) -> Result<DataFrame> {
    let mut col_name = Vec::with_capacity(columns.len());
    let mut n_rows_v = Vec::with_capacity(columns.len());
    let mut n_non_null_v = Vec::with_capacity(columns.len());
    let mut n_distinct_v = Vec::with_capacity(columns.len());
    let mut null_frac_v = Vec::with_capacity(columns.len());
    let mut distinct_frac_v = Vec::with_capacity(columns.len());

    for c in columns {
        let stats = lf
            .clone()
            .select([
                len().alias("n_rows"),
                col(*c).count().alias("n_non_null"),
                col(*c).drop_nulls().n_unique().alias("n_distinct"),
            ])
            .collect()
            .map_err(|e| WeldrsError::Training {
                stage: "exploratory",
                message: format!("Failed to profile '{c}': {e}"),
            })?;

        let n_rows = scalar_u64(&stats, "n_rows")?;
        let n_non_null = scalar_u64(&stats, "n_non_null")?;
        let n_distinct = scalar_u64(&stats, "n_distinct")?;

        col_name.push(c.to_string());
        n_rows_v.push(n_rows);
        n_non_null_v.push(n_non_null);
        n_distinct_v.push(n_distinct);
        null_frac_v.push(if n_rows == 0 {
            0.0
        } else {
            (n_rows - n_non_null) as f64 / n_rows as f64
        });
        distinct_frac_v.push(if n_non_null == 0 {
            0.0
        } else {
            n_distinct as f64 / n_non_null as f64
        });
    }

    DataFrame::new(
        columns.len(),
        vec![
            Column::new("column".into(), col_name),
            Column::new("n_rows".into(), n_rows_v),
            Column::new("n_non_null".into(), n_non_null_v),
            Column::new("n_distinct".into(), n_distinct_v),
            Column::new("null_fraction".into(), null_frac_v),
            Column::new("distinct_fraction".into(), distinct_frac_v),
        ],
    )
    .map_err(WeldrsError::Polars)
}

/// Per-column completeness: the fraction of non-null values.
///
/// Returns a DataFrame `[column, n_non_null, completeness]`.
///
/// # Errors
///
/// Returns an error if a column is missing or a Polars op fails.
pub fn completeness(lf: &LazyFrame, columns: &[&str]) -> Result<DataFrame> {
    let profile = profile_columns(lf, columns)?;
    let n_non_null = profile.column("n_non_null").map_err(WeldrsError::Polars)?;
    let null_fraction = profile
        .column("null_fraction")
        .map_err(WeldrsError::Polars)?;
    let completeness: Float64Chunked = null_fraction
        .f64()
        .map_err(WeldrsError::Polars)?
        .apply_values(|nf| 1.0 - nf);

    DataFrame::new(
        columns.len(),
        vec![
            Column::new(
                "column".into(),
                columns.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
            ),
            n_non_null.clone(),
            Column::new("completeness".into(), completeness.into_series()),
        ],
    )
    .map_err(WeldrsError::Polars)
}

/// The `top_n` most frequent values of a column, with counts and fractions.
///
/// Returns a DataFrame `[value, count, fraction]` sorted by descending count.
/// Null values are excluded.
///
/// # Errors
///
/// Returns an error if the column is missing or a Polars op fails.
pub fn value_frequencies(lf: &LazyFrame, column: &str, top_n: usize) -> Result<DataFrame> {
    lf.clone()
        .filter(col(column).is_not_null())
        .group_by([col(column).alias("value")])
        .agg([len().alias("count")])
        .with_column(
            (col("count").cast(DataType::Float64) / col("count").sum().cast(DataType::Float64))
                .alias("fraction"),
        )
        .sort(
            ["count"],
            SortMultipleOptions::default().with_order_descending(true),
        )
        .limit(top_n as u32)
        .collect()
        .map_err(|e| WeldrsError::Training {
            stage: "exploratory",
            message: format!("Failed to compute value frequencies for '{column}': {e}"),
        })
}

fn scalar_u64(df: &DataFrame, name: &str) -> Result<u64> {
    Ok(df
        .column(name)
        .map_err(WeldrsError::Polars)?
        .cast(&DataType::UInt64)
        .map_err(WeldrsError::Polars)?
        .u64()
        .map_err(WeldrsError::Polars)?
        .get(0)
        .unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lf() -> LazyFrame {
        df!(
            "city" => [Some("London"), Some("London"), Some("Paris"), None],
            "id" => [1i64, 2, 3, 4],
        )
        .unwrap()
        .lazy()
    }

    #[test]
    fn test_profile_columns() {
        let p = profile_columns(&lf(), &["city", "id"]).unwrap();
        assert_eq!(p.height(), 2);

        // city: 4 rows, 3 non-null, 2 distinct, null_fraction 0.25.
        let names: Vec<&str> = p
            .column("column")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let city_idx = names.iter().position(|c| *c == "city").unwrap();
        let null_frac = p.column("null_fraction").unwrap().f64().unwrap();
        assert!((null_frac.get(city_idx).unwrap() - 0.25).abs() < 1e-12);
        let n_distinct = p.column("n_distinct").unwrap().u64().unwrap();
        assert_eq!(n_distinct.get(city_idx), Some(2));
    }

    #[test]
    fn test_completeness() {
        let c = completeness(&lf(), &["city"]).unwrap();
        let comp = c.column("completeness").unwrap().f64().unwrap();
        assert!((comp.get(0).unwrap() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_value_frequencies() {
        let v = value_frequencies(&lf(), "city", 10).unwrap();
        // London (2) should be first.
        let value = v.column("value").unwrap().str().unwrap();
        let count = v.column("count").unwrap().u32().unwrap();
        assert_eq!(value.get(0), Some("London"));
        assert_eq!(count.get(0), Some(2));
    }
}
