use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::prelude::*;

fn people() -> DataFrame {
    df!(
        "unique_id" => [1i64, 2, 3, 4],
        "first_name" => ["John", "Jon", "Jane", "Mary"],
        "city" => ["London", "London", "Paris", "Paris"],
    )
    .unwrap()
}

/// A linker whose first_name comparison has known m/u so scores are deterministic.
fn trained_linker() -> Linker {
    let settings = Settings::builder(LinkType::DedupeOnly)
        .comparison(
            ComparisonBuilder::new("first_name")
                .null_level()
                .exact_match_level()
                .jaro_winkler_level(0.85)
                .else_level()
                .build()
                .unwrap(),
        )
        .blocking_rule(BlockingRule::on(&["city"]))
        .probability_two_random_records_match(0.1)
        .build()
        .unwrap();

    let mut linker = Linker::new(settings).unwrap();
    for level in &mut linker.settings_mut().comparisons[0].comparison_levels {
        if level.is_null_level {
            continue;
        }
        match level.comparison_vector_value {
            2 => {
                level.m_probability = Some(0.9);
                level.u_probability = Some(0.02);
            } // exact
            1 => {
                level.m_probability = Some(0.08);
                level.u_probability = Some(0.08);
            } // jaro-winkler
            _ => {
                level.m_probability = Some(0.02);
                level.u_probability = Some(0.9);
            } // else
        }
    }
    linker
}

#[test]
fn test_deterministic_link_returns_certain_pairs() {
    let linker = trained_linker();
    let lf = people().lazy();
    let result = linker
        .deterministic_link(&lf, &[BlockingRule::on(&["first_name"])])
        .unwrap();

    // No two people share an exact first_name here, so the block yields none;
    // block on city instead to get pairs.
    assert_eq!(result.height(), 0);

    let result = linker
        .deterministic_link(&lf, &[BlockingRule::on(&["city"])])
        .unwrap();
    assert!(result.height() > 0);
    // Every deterministic pair is certain.
    let probs = result.column("match_probability").unwrap().f64().unwrap();
    for p in probs.into_no_null_iter() {
        assert!((p - 1.0).abs() < 1e-12);
    }
}

#[test]
fn test_compare_two_records_strong_beats_weak() {
    let linker = trained_linker();

    let strong = linker
        .compare_two_records(
            &[("first_name", AnyValue::String("John"))],
            &[("first_name", AnyValue::String("John"))],
        )
        .unwrap();
    let weak = linker
        .compare_two_records(
            &[("first_name", AnyValue::String("John"))],
            &[("first_name", AnyValue::String("Zoltan"))],
        )
        .unwrap();

    assert_eq!(strong.height(), 1);
    assert_eq!(weak.height(), 1);

    let sw = strong
        .column("match_weight")
        .unwrap()
        .f64()
        .unwrap()
        .get(0)
        .unwrap();
    let ww = weak
        .column("match_weight")
        .unwrap()
        .f64()
        .unwrap()
        .get(0)
        .unwrap();
    assert!(
        sw > ww,
        "exact-match pair ({sw}) should score above a mismatch ({ww})"
    );
}

#[test]
fn test_find_matches_to_new_records() {
    let linker = trained_linker();
    let existing = people().lazy();
    let new = df!(
        "unique_id" => [101i64],
        "first_name" => ["John"],
        "city" => ["London"],
    )
    .unwrap()
    .lazy();

    let matches = linker
        .find_matches_to_new_records(&existing, &new, None)
        .unwrap()
        .collect()
        .unwrap();

    // The new "John in London" blocks against the two London records (1, 2).
    assert!(matches.height() > 0);
    let names: Vec<&str> = matches
        .get_column_names()
        .into_iter()
        .map(|s| s.as_str())
        .collect();
    assert!(names.contains(&"match_weight"));
    assert!(names.contains(&"match_probability"));

    // The exact John↔John pair should score above the John↔Jon pair.
    let uid_r = matches.column("unique_id_r").unwrap().i64().unwrap();
    let uid_l = matches.column("unique_id_l").unwrap().i64().unwrap();
    let mw = matches.column("match_weight").unwrap().f64().unwrap();
    let mut john = f64::MIN;
    let mut jon = f64::MIN;
    for ((l, r), w) in uid_l.into_iter().zip(uid_r).zip(mw) {
        let (l, r, w) = (l.unwrap(), r.unwrap(), w.unwrap());
        // new record has id 101; existing John=1, Jon=2.
        if l == 1 || r == 1 {
            john = john.max(w);
        }
        if l == 2 || r == 2 {
            jon = jon.max(w);
        }
    }
    assert!(
        john > jon,
        "John↔John ({john}) should beat John↔Jon ({jon})"
    );
}
