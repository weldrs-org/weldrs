//! Graphical chart rendering for model explanations.
//!
//! Provides SVG-based visualizations of waterfall charts, match weight
//! charts, and weight distribution histograms using the `plotters` crate.
//!
//! This module is feature-gated behind the `visualize` feature flag.
//! Enable it in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! weldrs = { version = "0.1", features = ["visualize"] }
//! ```
//!
//! Three chart types are available:
//!
//! - [`waterfall_chart_svg`] / [`waterfall_chart_to_file`] — step-by-step
//!   breakdown of a single pair's score (from [`explain`](crate::explain)).
//! - [`match_weights_chart_svg`] / [`match_weights_chart_to_file`] — bar
//!   chart of all trained comparison-level match weights.
//! - [`weight_distribution_chart_svg`] / [`weight_distribution_chart_to_file`]
//!   — histogram of match weights across all scored pairs.
//!
//! All rendering functions accept a [`ChartOptions`] struct for
//! customizing dimensions, colors, and font sizes.

use std::path::Path;

use plotters::prelude::*;

use crate::error::{Result, WeldrsError};
use crate::explain::{ModelSummary, WaterfallChart};

// ── Configuration ────────────────────────────────────────────────────

/// Options controlling chart appearance.
#[derive(Debug, Clone)]
pub struct ChartOptions {
    /// Width in pixels. Default: 900.
    pub width: u32,
    /// Height in pixels. Default: 500.
    pub height: u32,
    /// RGB color for positive (match evidence) bars. Default: green.
    pub positive_color: (u8, u8, u8),
    /// RGB color for negative (non-match evidence) bars. Default: red.
    pub negative_color: (u8, u8, u8),
    /// RGB color for the prior bar. Default: blue.
    pub prior_color: (u8, u8, u8),
    /// RGB color for neutral / histogram bars. Default: gray.
    pub neutral_color: (u8, u8, u8),
    /// Title font size. Default: 20.
    pub title_font_size: u32,
    /// Axis / label font size. Default: 14.
    pub label_font_size: u32,
}

impl Default for ChartOptions {
    fn default() -> Self {
        Self {
            width: 900,
            height: 500,
            positive_color: (76, 175, 80),  // green
            negative_color: (244, 67, 54),  // red
            prior_color: (33, 150, 243),    // blue
            neutral_color: (158, 158, 158), // gray
            title_font_size: 20,
            label_font_size: 14,
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

fn rgb(c: (u8, u8, u8)) -> RGBColor {
    RGBColor(c.0, c.1, c.2)
}

fn vis_err(msg: impl Into<String>) -> WeldrsError {
    WeldrsError::Visualization(msg.into())
}

fn write_svg_to_file(svg: &str, path: &Path) -> Result<()> {
    std::fs::write(path, svg).map_err(|e| vis_err(format!("Failed to write SVG: {e}")))
}

// ── 1. Waterfall chart ──────────────────────────────────────────────

/// Render a waterfall chart to an SVG string.
///
/// Each step in the waterfall becomes a floating bar colored green
/// (positive evidence) or red (negative evidence). The prior bar is
/// shown in blue. A connector line traces the cumulative match weight.
pub fn waterfall_chart_svg(waterfall: &WaterfallChart, options: &ChartOptions) -> Result<String> {
    let steps = &waterfall.steps;
    if steps.is_empty() {
        return Err(vis_err("Waterfall has no steps"));
    }

    let n = steps.len();

    // Compute bar extents: each bar goes from cumulative_before to cumulative_after.
    let mut bar_bottoms: Vec<f64> = Vec::with_capacity(n);
    let mut bar_tops: Vec<f64> = Vec::with_capacity(n);

    for (i, step) in steps.iter().enumerate() {
        if i == 0 {
            // Prior: bar from 0 to its weight.
            bar_bottoms.push(0.0);
            bar_tops.push(step.cumulative_match_weight);
        } else {
            let prev_cumulative = steps[i - 1].cumulative_match_weight;
            bar_bottoms.push(prev_cumulative);
            bar_tops.push(step.cumulative_match_weight);
        }
    }

    // Y-axis range with some padding.
    let y_min = bar_bottoms
        .iter()
        .chain(bar_tops.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let y_max = bar_bottoms
        .iter()
        .chain(bar_tops.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_range = y_max - y_min;
    let y_pad = if y_range.abs() < 1e-10 {
        1.0
    } else {
        y_range * 0.15
    };

    // Labels for x-axis.
    let labels: Vec<String> = steps.iter().map(|s| s.column_name.clone()).collect();

    let mut buf = String::new();
    {
        let root =
            SVGBackend::with_string(&mut buf, (options.width, options.height)).into_drawing_area();
        root.fill(&WHITE).map_err(|e| vis_err(e.to_string()))?;

        let title = format!(
            "Waterfall: pair ({}, {})",
            waterfall.unique_id_l, waterfall.unique_id_r
        );

        let mut chart = ChartBuilder::on(&root)
            .caption(&title, ("sans-serif", options.title_font_size))
            .margin(15)
            .x_label_area_size(60)
            .y_label_area_size(60)
            .build_cartesian_2d(
                (0..n - 1).into_segmented(),
                (y_min - y_pad)..(y_max + y_pad),
            )
            .map_err(|e| vis_err(e.to_string()))?;

        chart
            .configure_mesh()
            .x_labels(n)
            .x_label_formatter(&|seg| {
                if let SegmentValue::CenterOf(idx) = seg {
                    labels.get(*idx).cloned().unwrap_or_default()
                } else {
                    String::new()
                }
            })
            .x_label_style(("sans-serif", options.label_font_size).into_text_style(&root))
            .y_desc("Match weight")
            .y_label_style(("sans-serif", options.label_font_size).into_text_style(&root))
            .draw()
            .map_err(|e| vis_err(e.to_string()))?;

        // Draw bars.
        for i in 0..n {
            let bottom = bar_bottoms[i];
            let top = bar_tops[i];
            let color = if i == 0 {
                rgb(options.prior_color)
            } else if top >= bottom {
                rgb(options.positive_color)
            } else {
                rgb(options.negative_color)
            };

            let lo = bottom.min(top);
            let hi = bottom.max(top);

            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [
                        (SegmentValue::Exact(i), lo),
                        (SegmentValue::Exact(i + 1), hi),
                    ],
                    color.filled(),
                )))
                .map_err(|e| vis_err(e.to_string()))?;

            // Weight label on each bar.
            let weight = steps[i].log2_bayes_factor;
            let label_text = format!("{weight:+.2}");
            let label_y = hi + y_pad * 0.08;
            chart
                .draw_series(std::iter::once(Text::new(
                    label_text,
                    (SegmentValue::CenterOf(i), label_y),
                    ("sans-serif", 11).into_font(),
                )))
                .map_err(|e| vis_err(e.to_string()))?;
        }

        // Connector line through cumulative midpoints.
        let connector: Vec<(SegmentValue<usize>, f64)> = steps
            .iter()
            .enumerate()
            .map(|(i, s)| (SegmentValue::CenterOf(i), s.cumulative_match_weight))
            .collect();
        chart
            .draw_series(LineSeries::new(connector, &BLACK))
            .map_err(|e| vis_err(e.to_string()))?;

        root.present().map_err(|e| vis_err(e.to_string()))?;
    }

    Ok(buf)
}

/// Render a waterfall chart and write the SVG to a file.
pub fn waterfall_chart_to_file(
    waterfall: &WaterfallChart,
    path: &Path,
    options: &ChartOptions,
) -> Result<()> {
    let svg = waterfall_chart_svg(waterfall, options)?;
    write_svg_to_file(&svg, path)
}

// ── 2. Match weights chart ──────────────────────────────────────────

/// Render a match weights chart to an SVG string.
///
/// Shows one bar per non-null comparison level, grouped by comparison
/// name and colored green/red by sign. Bars are anchored at y=0.
pub fn match_weights_chart_svg(summary: &ModelSummary, options: &ChartOptions) -> Result<String> {
    // Collect bars: (label, weight).
    let mut bar_labels: Vec<String> = Vec::new();
    let mut bar_weights: Vec<f64> = Vec::new();

    // Add prior bar.
    bar_labels.push("Prior".to_string());
    bar_weights.push(summary.prior_match_weight);

    for comp in &summary.comparisons {
        for level in &comp.levels {
            if level.is_null_level {
                continue;
            }
            if let Some(w) = level.log2_bayes_factor {
                let label = format!("{}\n{}", comp.output_column_name, level.label);
                bar_labels.push(label);
                bar_weights.push(w);
            }
        }
    }

    let n = bar_labels.len();
    if n == 0 {
        return Err(vis_err("No levels to display"));
    }

    let y_min = bar_weights.iter().cloned().fold(0.0_f64, f64::min);
    let y_max = bar_weights.iter().cloned().fold(0.0_f64, f64::max);
    let y_range = y_max - y_min;
    let y_pad = if y_range.abs() < 1e-10 {
        1.0
    } else {
        y_range * 0.15
    };

    let mut buf = String::new();
    {
        let root =
            SVGBackend::with_string(&mut buf, (options.width, options.height)).into_drawing_area();
        root.fill(&WHITE).map_err(|e| vis_err(e.to_string()))?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Match weights by comparison level",
                ("sans-serif", options.title_font_size),
            )
            .margin(15)
            .x_label_area_size(80)
            .y_label_area_size(60)
            .build_cartesian_2d(
                (0..n - 1).into_segmented(),
                (y_min - y_pad)..(y_max + y_pad),
            )
            .map_err(|e| vis_err(e.to_string()))?;

        chart
            .configure_mesh()
            .x_labels(n)
            .x_label_formatter(&|seg| {
                if let SegmentValue::CenterOf(idx) = seg {
                    // Return only the first line (comparison name) for the axis.
                    bar_labels
                        .get(*idx)
                        .map(|l| l.lines().next().unwrap_or("").to_string())
                        .unwrap_or_default()
                } else {
                    String::new()
                }
            })
            .x_label_style(("sans-serif", options.label_font_size).into_text_style(&root))
            .y_desc("Match weight (log2 BF)")
            .y_label_style(("sans-serif", options.label_font_size).into_text_style(&root))
            .draw()
            .map_err(|e| vis_err(e.to_string()))?;

        for (i, &w) in bar_weights.iter().enumerate().take(n) {
            let color = if i == 0 {
                rgb(options.prior_color)
            } else if w >= 0.0 {
                rgb(options.positive_color)
            } else {
                rgb(options.negative_color)
            };

            let (lo, hi) = if w >= 0.0 { (0.0, w) } else { (w, 0.0) };

            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [
                        (SegmentValue::Exact(i), lo),
                        (SegmentValue::Exact(i + 1), hi),
                    ],
                    color.filled(),
                )))
                .map_err(|e| vis_err(e.to_string()))?;

            // Level label beneath the bar.
            let level_label = bar_labels
                .get(i)
                .map(|l| l.lines().nth(1).unwrap_or("").to_string())
                .unwrap_or_default();
            if !level_label.is_empty() {
                let label_y = if w >= 0.0 {
                    hi + y_pad * 0.08
                } else {
                    lo - y_pad * 0.08
                };
                chart
                    .draw_series(std::iter::once(Text::new(
                        level_label,
                        (SegmentValue::CenterOf(i), label_y),
                        ("sans-serif", 10).into_font(),
                    )))
                    .map_err(|e| vis_err(e.to_string()))?;
            }
        }

        root.present().map_err(|e| vis_err(e.to_string()))?;
    }

    Ok(buf)
}

/// Render a match weights chart and write the SVG to a file.
pub fn match_weights_chart_to_file(
    summary: &ModelSummary,
    path: &Path,
    options: &ChartOptions,
) -> Result<()> {
    let svg = match_weights_chart_svg(summary, options)?;
    write_svg_to_file(&svg, path)
}

// ── 3. Weight distribution histogram ────────────────────────────────

/// Render a histogram of match weight values to an SVG string.
///
/// Performs manual binning of the supplied match weights and draws
/// rectangular bars for each bin.
pub fn weight_distribution_chart_svg(
    match_weights: &[f64],
    num_bins: Option<usize>,
    options: &ChartOptions,
) -> Result<String> {
    if match_weights.is_empty() {
        return Err(vis_err("No match weights to plot"));
    }

    let finite: Vec<f64> = match_weights
        .iter()
        .copied()
        .filter(|w| w.is_finite())
        .collect();
    if finite.is_empty() {
        return Err(vis_err("All match weights are non-finite"));
    }

    let w_min = finite.iter().cloned().fold(f64::INFINITY, f64::min);
    let w_max = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let bins = num_bins.unwrap_or_else(|| {
        // Sturges' rule.
        let k = ((finite.len() as f64).log2().ceil() as usize) + 1;
        k.clamp(5, 50)
    });

    let range = w_max - w_min;
    let bin_width = if range.abs() < 1e-10 {
        1.0
    } else {
        range / bins as f64
    };

    let mut counts = vec![0u32; bins];
    for &w in &finite {
        let idx = ((w - w_min) / bin_width) as usize;
        let idx = idx.min(bins - 1);
        counts[idx] += 1;
    }

    let max_count = *counts.iter().max().unwrap_or(&1);

    let mut buf = String::new();
    {
        let root =
            SVGBackend::with_string(&mut buf, (options.width, options.height)).into_drawing_area();
        root.fill(&WHITE).map_err(|e| vis_err(e.to_string()))?;

        let x_pad = bin_width * 0.5;
        let y_pad = max_count as f64 * 0.1;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Match weight distribution",
                ("sans-serif", options.title_font_size),
            )
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                (w_min - x_pad)..(w_max + x_pad),
                0.0..(max_count as f64 + y_pad),
            )
            .map_err(|e| vis_err(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc("Match weight")
            .y_desc("Count")
            .x_label_style(("sans-serif", options.label_font_size).into_text_style(&root))
            .y_label_style(("sans-serif", options.label_font_size).into_text_style(&root))
            .draw()
            .map_err(|e| vis_err(e.to_string()))?;

        let bar_color = rgb(options.neutral_color);
        for (i, &count) in counts.iter().enumerate() {
            let x0 = w_min + i as f64 * bin_width;
            let x1 = x0 + bin_width;
            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(x0, 0.0), (x1, count as f64)],
                    bar_color.filled(),
                )))
                .map_err(|e| vis_err(e.to_string()))?;
        }

        root.present().map_err(|e| vis_err(e.to_string()))?;
    }

    Ok(buf)
}

/// Render a weight distribution histogram and write the SVG to a file.
pub fn weight_distribution_chart_to_file(
    match_weights: &[f64],
    num_bins: Option<usize>,
    path: &Path,
    options: &ChartOptions,
) -> Result<()> {
    let svg = weight_distribution_chart_svg(match_weights, num_bins, options)?;
    write_svg_to_file(&svg, path)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocking::BlockingRule;
    use crate::comparison::ComparisonBuilder;
    use crate::explain;
    use crate::predict;
    use crate::settings::{LinkType, Settings};
    use polars::prelude::IntoLazy;

    fn make_trained_settings() -> Settings {
        let mut comp_fn = ComparisonBuilder::new("first_name")
            .null_level()
            .exact_match_level()
            .else_level()
            .build();
        for level in &mut comp_fn.comparison_levels {
            if level.is_null_level {
                continue;
            }
            if level.comparison_vector_value == 1 {
                level.m_probability = Some(0.9);
                level.u_probability = Some(0.1);
            } else {
                level.m_probability = Some(0.1);
                level.u_probability = Some(0.9);
            }
        }

        let mut comp_sn = ComparisonBuilder::new("last_name")
            .null_level()
            .exact_match_level()
            .else_level()
            .build();
        for level in &mut comp_sn.comparison_levels {
            if level.is_null_level {
                continue;
            }
            if level.comparison_vector_value == 1 {
                level.m_probability = Some(0.85);
                level.u_probability = Some(0.05);
            } else {
                level.m_probability = Some(0.15);
                level.u_probability = Some(0.95);
            }
        }

        Settings::builder(LinkType::DedupeOnly)
            .comparison(comp_fn)
            .comparison(comp_sn)
            .probability_two_random_records_match(0.01)
            .blocking_rule(BlockingRule::on(&["last_name"]))
            .build()
            .unwrap()
    }

    fn make_waterfall(settings: &Settings) -> WaterfallChart {
        let df = polars::prelude::df!(
            "unique_id_l" => [1i64],
            "unique_id_r" => [2i64],
            "first_name_l" => ["Alice"],
            "first_name_r" => ["Alice"],
            "last_name_l" => ["Smith"],
            "last_name_r" => ["Smith"],
            "gamma_first_name" => [1i8],
            "gamma_last_name" => [1i8],
        )
        .unwrap();

        let predictions = predict::predict(
            df.lazy(),
            &settings.comparisons,
            settings.probability_two_random_records_match,
            &settings.gamma_prefix,
            &settings.bf_prefix,
            None,
            None,
        )
        .unwrap()
        .collect()
        .unwrap();

        explain::explain_pair(
            &predictions,
            0,
            &settings.comparisons,
            settings.probability_two_random_records_match,
            &settings.gamma_prefix,
            &settings.bf_prefix,
            &settings.unique_id_column,
        )
        .unwrap()
    }

    #[test]
    fn test_waterfall_chart_produces_svg() {
        let settings = make_trained_settings();
        let waterfall = make_waterfall(&settings);
        let svg = waterfall_chart_svg(&waterfall, &ChartOptions::default()).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("Waterfall"));
    }

    #[test]
    fn test_waterfall_chart_empty_steps_errors() {
        let wf = WaterfallChart {
            unique_id_l: "a".into(),
            unique_id_r: "b".into(),
            steps: vec![],
            final_match_weight: 0.0,
            final_match_probability: 0.5,
        };
        assert!(waterfall_chart_svg(&wf, &ChartOptions::default()).is_err());
    }

    #[test]
    fn test_waterfall_chart_to_file() {
        let settings = make_trained_settings();
        let waterfall = make_waterfall(&settings);
        let dir = std::env::temp_dir();
        let path = dir.join("weldrs_test_waterfall.svg");
        waterfall_chart_to_file(&waterfall, &path, &ChartOptions::default()).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("<svg"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_match_weights_chart_produces_svg() {
        let settings = make_trained_settings();
        let summary = explain::model_summary(&settings);
        let svg = match_weights_chart_svg(&summary, &ChartOptions::default()).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("Match weights"));
    }

    #[test]
    fn test_match_weights_chart_to_file() {
        let settings = make_trained_settings();
        let summary = explain::model_summary(&settings);
        let dir = std::env::temp_dir();
        let path = dir.join("weldrs_test_match_weights.svg");
        match_weights_chart_to_file(&summary, &path, &ChartOptions::default()).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("<svg"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_histogram_produces_svg() {
        let weights = vec![-5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 5.0, 8.0, 10.0];
        let svg = weight_distribution_chart_svg(&weights, None, &ChartOptions::default()).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("distribution"));
    }

    #[test]
    fn test_histogram_custom_bins() {
        let weights = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let svg =
            weight_distribution_chart_svg(&weights, Some(3), &ChartOptions::default()).unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn test_histogram_empty_errors() {
        let weights: Vec<f64> = vec![];
        assert!(weight_distribution_chart_svg(&weights, None, &ChartOptions::default()).is_err());
    }

    #[test]
    fn test_histogram_all_nonfinite_errors() {
        let weights = vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
        assert!(weight_distribution_chart_svg(&weights, None, &ChartOptions::default()).is_err());
    }

    #[test]
    fn test_histogram_to_file() {
        let weights = vec![0.0, 1.0, 2.0, 3.0];
        let dir = std::env::temp_dir();
        let path = dir.join("weldrs_test_histogram.svg");
        weight_distribution_chart_to_file(&weights, None, &path, &ChartOptions::default()).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("<svg"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_chart_options_default() {
        let opts = ChartOptions::default();
        assert_eq!(opts.width, 900);
        assert_eq!(opts.height, 500);
        assert_eq!(opts.title_font_size, 20);
        assert_eq!(opts.label_font_size, 14);
    }

    #[test]
    fn test_custom_chart_options() {
        let opts = ChartOptions {
            width: 1200,
            height: 800,
            ..Default::default()
        };
        let weights = vec![0.0, 1.0, 2.0];
        let svg = weight_distribution_chart_svg(&weights, None, &opts).unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn test_histogram_single_value() {
        let weights = vec![5.0, 5.0, 5.0];
        let svg =
            weight_distribution_chart_svg(&weights, Some(5), &ChartOptions::default()).unwrap();
        assert!(svg.contains("<svg"));
    }
}
