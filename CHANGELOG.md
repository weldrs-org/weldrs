# Changelog

All notable changes to **weldrs** are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## 0.3.0

A large feature release that substantially closes the gap with the Python
[Splink](https://moj-analytical-services.github.io/splink/) package, plus a few
breaking changes batched into this version.

### Breaking changes

- **`Linker.settings` is now a private field.** Use `Linker::settings()` /
  `Linker::settings_mut()` (both already existed). The previously-deprecated
  public field has been removed.
- **`Linker::predict` now defaults to `PredictMode::Auto`** (was `Lazy`). Auto
  picks Direct for small candidate sets and Lazy for large ones. Use
  `predict_with_mode` to force a strategy.
- **EM now defaults to `fix_u_probabilities = true`** (matching Splink). The EM
  trainer no longer re-estimates u-probabilities by default — u comes from
  `estimate_u_using_random_sampling`. Use `estimate_parameters_using_em_with_options`
  with a custom `EmRunOptions` to change this.
- **`predict` / `predict_direct` and `explain_pair` / `explain_pairs` gained a
  `tf_prefix` parameter** (term-frequency adjustment column prefix).
- **`LinkType::LinkOnly` now requires a `source_dataset_column`**, validated at
  `Settings` build time.
- The free function `em::expectation_maximization` gained an `EmRunOptions`
  parameter.

### Added

- **Comparison library** — new predicates: Damerau-Levenshtein, Hamming,
  Jaccard (character-set), AbsoluteDateDifference (day/month/year),
  PercentageDifference, DistanceInKm (Haversine), ArrayIntersect,
  CosineSimilarity; plus `And` / `Or` / `Not` composition and `CustomPredicate`
  (Polars-SQL DSL). Matching `ComparisonBuilder` methods for each.
- **Term-frequency adjustments** — opt in per comparison via
  `ComparisonBuilder::with_term_frequency_adjustments` (with `tf_adjustment_weight`
  / `tf_minimum_u_value`). Rare agreeing values score higher. Wired through
  prediction and the waterfall (new `tf_adjustment` / `bayes_factor_pre_tf`
  fields on `WaterfallStep`).
- **EM controls** — `EmRunOptions` (`fix_m_probabilities`, `fix_u_probabilities`,
  `fix_probability_two_random_records_match`,
  `populate_probability_two_random_records_match_from_trained_values`) and
  `Linker::estimate_parameters_using_em_with_options`. The EM-trained lambda is
  now written back when requested (previously dropped).
- **Inference helpers** — `Linker::deterministic_link`,
  `compare_two_records`, and `find_matches_to_new_records`.
- **Labelled training & evaluation** — `estimate_m_from_label_column`,
  `estimate_m_from_pairwise_labels`, plus `accuracy_analysis`, `roc_table`,
  `precision_recall_table`, and `prediction_errors_from_labels` (new
  `evaluation` module + `ThresholdMetrics`).
- **Blocking** — `BlockingRule::custom` (SQL predicate) and `and` / `or` / `not`
  combinators (an AND of equi-joins stays a fast equi-join); new
  `blocking_analysis` module (`count_comparisons_from_blocking_rule`,
  `cumulative_comparisons_from_blocking_rules`, `n_largest_blocks`).
- **Clustering** — `cluster_using_single_best_links` (per-source cardinality
  constraint) and `compute_graph_metrics` (degree, bridge edges, cluster density)
  via `petgraph` (new `graph_metrics` module).
- **Settings retention** — `retain_matching_columns`,
  `retain_intermediate_calculation_columns`, `additional_columns_to_retain`.
- **Exploratory profiling** — new `exploratory` module (`profile_columns`,
  `completeness`, `value_frequencies`).
- **Diagnostics** (behind the `visualize` feature) — `m_u_parameters_chart_svg`,
  `tf_adjustment_chart_svg` (term-frequency multiplier histogram), and
  `parameter_estimate_comparisons_chart_svg` (m-probability trajectory across EM
  iterations).
- **`EmOutcome` return shape** — `expectation_maximization` and the `Linker` EM
  methods now return an `EmOutcome { final_result, history }` (history populated
  when `store_history` is enabled), replacing the previous `Vec<EmIterationResult>`.
- Per-iteration EM progress logging at `debug!`.

### Changed

- `estimate_u` refactored onto shared `training_common` helpers (also used by
  m-estimation), eliminating duplicated frequency logic.
- The whole crate is now `cargo clippy -D warnings` clean.

### Fixed

- **EM no longer biases m-estimates with the blocked column.** A comparison
  whose column is part of the EM training block is now *excluded* from the
  E-step (its Bayes factor is neutralized to 1) rather than contributing its
  default/previous value, which previously saturated the posterior and deflated
  the other comparisons' m-probabilities. Verified against Splink on the
  `fake_1000` dataset (e.g. first-name exact-match m moved from ~0.18 to ~0.41
  vs Splink's ~0.55; λ matches exactly).

### Notes / deferred

- A configurable unique-pair dedup threshold (the 50% heuristic in
  `par_pairwise_string_predicate`) remains hard-coded; exposing it cleanly would
  require threading config through expression construction, for low payoff.

## 0.2.2

- Prior release (Fellegi-Sunter core: comparisons, blocking, EM, predict,
  connected-components clustering, JSON model serialization, SVG waterfall &
  match-weight charts).
