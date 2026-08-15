use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::predict::PredictMode;
use weldrs::prelude::*;

/// Six records sharing a constant block key so every pair is generated.
/// Surname "Smith" is common (4×), "Zelenskyy" is rare (2×).
fn tf_dataset() -> DataFrame {
    df!(
        "unique_id" => [1i64, 2, 3, 4, 5, 6],
        "block" => ["x", "x", "x", "x", "x", "x"],
        "surname" => ["Smith", "Smith", "Smith", "Smith", "Zelenskyy", "Zelenskyy"],
    )
    .unwrap()
}

/// Build a linker whose single surname comparison has known m/u and (optionally)
/// term-frequency adjustments enabled.
fn make_linker(tf: bool) -> Linker {
    let mut builder = ComparisonBuilder::new("surname")
        .null_level()
        .exact_match_level()
        .else_level();
    if tf {
        builder = builder.with_term_frequency_adjustments();
    }
    let comparison = builder.build().unwrap();

    let settings = Settings::builder(LinkType::DedupeOnly)
        .comparison(comparison)
        .blocking_rule(BlockingRule::on(&["block"]))
        .probability_two_random_records_match(0.1)
        // Keep bf_/tf_ columns so the waterfall can show the TF split.
        .retain_intermediate_calculation_columns(true)
        .build()
        .unwrap();

    let mut linker = Linker::new(settings).unwrap();
    // Set deterministic m/u: exact m=0.9 u=0.1, else m=0.1 u=0.9.
    for level in &mut linker.settings_mut().comparisons[0].comparison_levels {
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
    linker
}

/// Extract the match_weight for a specific (uid_l, uid_r) pair.
fn weight_for(predictions: &DataFrame, a: i64, b: i64) -> f64 {
    let uid_l = predictions.column("unique_id_l").unwrap().i64().unwrap();
    let uid_r = predictions.column("unique_id_r").unwrap().i64().unwrap();
    let mw = predictions.column("match_weight").unwrap().f64().unwrap();
    for ((l, r), w) in uid_l.into_iter().zip(uid_r).zip(mw) {
        let (l, r, w) = (l.unwrap(), r.unwrap(), w.unwrap());
        if (l, r) == (a, b) || (l, r) == (b, a) {
            return w;
        }
    }
    panic!("pair ({a},{b}) not found in predictions");
}

#[test]
fn test_tf_upweights_rare_value_matches() {
    let lf = tf_dataset().lazy();
    let linker = make_linker(true);
    let predictions = linker.predict(&lf, None).unwrap().collect().unwrap();

    // Both are exact-surname matches, but Zelenskyy is rarer than Smith, so its
    // match weight must be strictly higher under term-frequency adjustment.
    let smith = weight_for(&predictions, 1, 2);
    let zelenskyy = weight_for(&predictions, 5, 6);
    assert!(
        zelenskyy > smith,
        "rare-value match should score higher: zelenskyy={zelenskyy}, smith={smith}"
    );
}

#[test]
fn test_tf_disabled_scores_equal_for_same_gamma() {
    let lf = tf_dataset().lazy();
    let linker = make_linker(false);
    let predictions = linker.predict(&lf, None).unwrap().collect().unwrap();

    // Without TF, both exact matches get identical weights regardless of value.
    let smith = weight_for(&predictions, 1, 2);
    let zelenskyy = weight_for(&predictions, 5, 6);
    assert!(
        (smith - zelenskyy).abs() < 1e-12,
        "without TF, same-gamma pairs should score equally: {smith} vs {zelenskyy}"
    );
}

#[test]
fn test_tf_lazy_and_direct_agree() {
    let lf = tf_dataset().lazy();
    let linker = make_linker(true);

    let lazy = linker
        .predict_with_mode(&lf, None, PredictMode::Lazy)
        .unwrap()
        .collect()
        .unwrap();
    let direct = linker
        .predict_with_mode(&lf, None, PredictMode::Direct)
        .unwrap()
        .collect()
        .unwrap();

    let lazy_probs: Vec<f64> = lazy
        .column("match_probability")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let direct_probs: Vec<f64> = direct
        .column("match_probability")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(lazy_probs.len(), direct_probs.len());
    for (l, d) in lazy_probs.iter().zip(direct_probs.iter()) {
        assert!(
            (l - d).abs() < 1e-12,
            "lazy/direct mismatch under TF: {l} vs {d}"
        );
    }
}

#[test]
fn test_tf_waterfall_exposes_split_and_stays_consistent() {
    let lf = tf_dataset().lazy();
    let linker = make_linker(true);
    let predictions = linker.predict(&lf, None).unwrap().collect().unwrap();

    // Explain row 0 and find the surname step.
    let chart = linker.explain_pair(&predictions, 0).unwrap();
    let surname_step = chart
        .steps
        .iter()
        .find(|s| s.column_name == "surname")
        .expect("surname step present");

    // TF split is populated and internally consistent.
    let tf = surname_step.tf_adjustment.expect("tf_adjustment present");
    let pre = surname_step
        .bayes_factor_pre_tf
        .expect("bayes_factor_pre_tf present");
    assert!(
        (surname_step.bayes_factor - pre * tf).abs() < 1e-9,
        "bayes_factor ({}) should equal pre_tf ({pre}) * tf ({tf})",
        surname_step.bayes_factor
    );

    // The waterfall's final weight matches the prediction's match_weight.
    let mw = predictions
        .column("match_weight")
        .unwrap()
        .f64()
        .unwrap()
        .get(0)
        .unwrap();
    assert!((chart.final_match_weight - mw).abs() < 1e-9);
}

#[test]
fn test_tf_settings_serde_roundtrip() {
    let comparison = ComparisonBuilder::new("surname")
        .null_level()
        .exact_match_level()
        .else_level()
        .with_term_frequency_adjustments()
        .tf_adjustment_weight(0.5)
        .build()
        .unwrap();
    let settings = Settings::builder(LinkType::DedupeOnly)
        .comparison(comparison)
        .blocking_rule(BlockingRule::on(&["block"]))
        .build()
        .unwrap();
    let linker = Linker::new(settings).unwrap();

    let json = linker.save_settings_json().unwrap();
    let restored = Linker::load_settings_json(&json).unwrap();
    let comp = &restored.settings().comparisons[0];
    assert!(comp.term_frequency_adjustments);
    assert!((comp.tf_adjustment_weight - 0.5).abs() < 1e-12);
    assert_eq!(restored.settings().tf_adjustment_column_prefix, "tf_");
}
