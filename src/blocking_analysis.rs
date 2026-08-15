//! Blocking-rule analysis.
//!
//! Tools for sizing and tuning blocking rules before prediction: how many
//! candidate pairs a rule produces, how the total grows as rules are combined,
//! and which blocks are largest (skew detection). These guide the trade-off
//! between recall (more pairs) and cost (quadratic blow-up).

use polars::prelude::*;

use crate::blocking::{self, BlockingRule};
use crate::error::{Result, WeldrsError};
use crate::settings::LinkType;

/// Count the candidate pairs a single blocking rule would generate.
///
/// # Errors
///
/// Returns an error if blocking or the count query fails.
pub fn count_comparisons_from_blocking_rule(
    lf: &LazyFrame,
    rule: &BlockingRule,
    link_type: &LinkType,
    unique_id_col: &str,
    source_dataset_column: Option<&str>,
) -> Result<u64> {
    let blocked = blocking::generate_blocked_pairs(
        lf,
        std::slice::from_ref(rule),
        link_type,
        unique_id_col,
        source_dataset_column,
    )?;
    count_rows(blocked, "blocking_analysis")
}

/// For a sequence of blocking rules, report how many *new* candidate pairs each
/// rule contributes (after de-duplicating against all earlier rules) and the
/// running cumulative total.
///
/// Returns a DataFrame with columns `rule_index` (u32), `description` (str),
/// `new_comparisons` (u64), and `cumulative_comparisons` (u64).
///
/// # Errors
///
/// Returns an error if blocking or a Polars op fails.
pub fn cumulative_comparisons_from_blocking_rules(
    lf: &LazyFrame,
    rules: &[BlockingRule],
    link_type: &LinkType,
    unique_id_col: &str,
    source_dataset_column: Option<&str>,
) -> Result<DataFrame> {
    let uid_l = format!("{unique_id_col}_l");
    let uid_r = format!("{unique_id_col}_r");

    let mut accumulated: Option<LazyFrame> = None;
    let mut rule_index: Vec<u32> = Vec::with_capacity(rules.len());
    let mut description: Vec<String> = Vec::with_capacity(rules.len());
    let mut new_comparisons: Vec<u64> = Vec::with_capacity(rules.len());
    let mut cumulative_comparisons: Vec<u64> = Vec::with_capacity(rules.len());
    let mut cumulative = 0u64;

    for (i, rule) in rules.iter().enumerate() {
        let pairs = blocking::generate_blocked_pairs(
            lf,
            std::slice::from_ref(rule),
            link_type,
            unique_id_col,
            source_dataset_column,
        )?
        .select([col(uid_l.as_str()), col(uid_r.as_str())]);

        // Pairs not already produced by an earlier rule.
        let new_pairs = match &accumulated {
            None => pairs.clone(),
            Some(acc) => pairs.clone().join(
                acc.clone(),
                [col(uid_l.as_str()), col(uid_r.as_str())],
                [col(uid_l.as_str()), col(uid_r.as_str())],
                JoinArgs::new(JoinType::Anti),
            ),
        };

        let n_new = count_rows(new_pairs.clone(), "blocking_analysis")?;
        cumulative += n_new;

        accumulated = Some(match accumulated {
            None => pairs,
            Some(acc) => concat([acc, new_pairs], UnionArgs::default())?,
        });

        rule_index.push(i as u32);
        description.push(
            rule.description
                .clone()
                .unwrap_or_else(|| format!("rule {i}")),
        );
        new_comparisons.push(n_new);
        cumulative_comparisons.push(cumulative);
    }

    DataFrame::new(
        rules.len(),
        vec![
            Column::new("rule_index".into(), rule_index),
            Column::new("description".into(), description),
            Column::new("new_comparisons".into(), new_comparisons),
            Column::new("cumulative_comparisons".into(), cumulative_comparisons),
        ],
    )
    .map_err(WeldrsError::Polars)
}

/// Report the `n` largest blocks produced by an equi-join blocking rule.
///
/// Groups the input by the rule's columns and returns, for each of the `n`
/// largest blocks, the block key column(s), the `block_size` (record count),
/// and `pairs` (`size * (size - 1) / 2`, the candidate pairs that block alone
/// would generate). Useful for spotting skew (e.g. a single value that blows up
/// the comparison count).
///
/// # Errors
///
/// Returns an error if the rule has no columns (a custom-predicate rule), or if
/// a Polars op fails.
pub fn n_largest_blocks(lf: &LazyFrame, rule: &BlockingRule, n: usize) -> Result<DataFrame> {
    if rule.columns.is_empty() {
        return Err(WeldrsError::Config(
            "n_largest_blocks requires an equi-join blocking rule with at least one column".into(),
        ));
    }

    let group_cols: Vec<Expr> = rule.columns.iter().map(|c| col(c.as_str())).collect();
    let size = col("block_size").cast(DataType::Float64);
    lf.clone()
        .group_by(group_cols)
        .agg([len().alias("block_size")])
        .sort(
            ["block_size"],
            SortMultipleOptions::default().with_order_descending(true),
        )
        .limit(n as u32)
        .with_column(
            ((size.clone() * (size - lit(1.0))) / lit(2.0))
                .cast(DataType::UInt64)
                .alias("pairs"),
        )
        .collect()
        .map_err(WeldrsError::Polars)
}

/// Collect a LazyFrame and return its row count.
fn count_rows(lf: LazyFrame, stage: &'static str) -> Result<u64> {
    let df = lf
        .select([len().alias("__n")])
        .collect()
        .map_err(|e| WeldrsError::Training {
            stage,
            message: format!("Count query failed: {e}"),
        })?;
    let n = df
        .column("__n")
        .map_err(WeldrsError::Polars)?
        .cast(&DataType::UInt64)
        .map_err(WeldrsError::Polars)?
        .u64()
        .map_err(WeldrsError::Polars)?
        .get(0)
        .unwrap_or(0);
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lf() -> LazyFrame {
        df!(
            "unique_id" => [1i64, 2, 3, 4, 5],
            "city" => ["London", "London", "London", "Paris", "Paris"],
            "first_name" => ["A", "B", "C", "A", "B"],
        )
        .unwrap()
        .lazy()
    }

    #[test]
    fn test_count_comparisons() {
        // London block has 3 records → 3 pairs; Paris has 2 → 1 pair. Total 4.
        let n = count_comparisons_from_blocking_rule(
            &lf(),
            &BlockingRule::on(&["city"]),
            &LinkType::DedupeOnly,
            "unique_id",
            None,
        )
        .unwrap();
        assert_eq!(n, 4);
    }

    #[test]
    fn test_cumulative_comparisons_dedup() {
        let rules = vec![
            BlockingRule::on(&["city"]).with_description("by city"),
            BlockingRule::on(&["first_name"]).with_description("by first_name"),
        ];
        let out = cumulative_comparisons_from_blocking_rules(
            &lf(),
            &rules,
            &LinkType::DedupeOnly,
            "unique_id",
            None,
        )
        .unwrap();

        assert_eq!(out.height(), 2);
        let cumulative = out.column("cumulative_comparisons").unwrap().u64().unwrap();
        // Rule 0 (city) → 4 pairs. Rule 1 (first_name): A={1,4},B={2,5} → (1,4),(2,5)
        // neither shares a city, so both new → cumulative 6.
        assert_eq!(cumulative.get(0), Some(4));
        assert_eq!(cumulative.get(1), Some(6));
    }

    #[test]
    fn test_n_largest_blocks() {
        let out = n_largest_blocks(&lf(), &BlockingRule::on(&["city"]), 1).unwrap();
        assert_eq!(out.height(), 1);
        // Largest block is London (3 records → 3 pairs).
        let size = out
            .column("block_size")
            .unwrap()
            .cast(&DataType::UInt64)
            .unwrap();
        assert_eq!(size.u64().unwrap().get(0), Some(3));
        let pairs = out.column("pairs").unwrap().u64().unwrap();
        assert_eq!(pairs.get(0), Some(3));
    }

    #[test]
    fn test_n_largest_blocks_requires_columns() {
        let rule = BlockingRule::custom("city_l = city_r");
        assert!(n_largest_blocks(&lf(), &rule, 1).is_err());
    }
}
