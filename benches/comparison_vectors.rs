use criterion::{Criterion, criterion_group, criterion_main};
use polars::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use weldrs::comparison::ComparisonBuilder;

const FIRST_NAMES: &[&str] = &[
    "James",
    "Mary",
    "Robert",
    "Patricia",
    "John",
    "Jennifer",
    "Michael",
    "Linda",
    "David",
    "Elizabeth",
    "William",
    "Barbara",
    "Richard",
    "Susan",
    "Joseph",
    "Jessica",
    "Thomas",
    "Sarah",
    "Charles",
    "Karen",
    "Daniel",
    "Lisa",
    "Matthew",
    "Nancy",
    "Anthony",
    "Betty",
    "Mark",
    "Margaret",
    "Donald",
    "Sandra",
];

/// Build a blocked-pairs DataFrame with controlled duplication.
///
/// `n_pairs` rows are generated, with left/right name values sampled from
/// a pool of `n_unique_names` distinct names. When `n_unique_names` is small
/// relative to `n_pairs`, many pairs share the same string values, letting
/// the dedup optimisation avoid redundant distance computations.
fn make_blocked_pairs(n_pairs: usize, n_unique_names: usize) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(42);

    let pool: Vec<String> = if n_unique_names <= FIRST_NAMES.len() {
        FIRST_NAMES[..n_unique_names]
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        (0..n_unique_names)
            .map(|i| format!("name_{i:05}"))
            .collect()
    };

    let uid_l: Vec<i64> = (0..n_pairs as i64).collect();
    let uid_r: Vec<i64> = (n_pairs as i64..2 * n_pairs as i64).collect();
    let name_l: Vec<&str> = (0..n_pairs)
        .map(|_| pool[rng.gen_range(0..pool.len())].as_str())
        .collect();
    let name_r: Vec<&str> = (0..n_pairs)
        .map(|_| pool[rng.gen_range(0..pool.len())].as_str())
        .collect();

    df!(
        "unique_id_l" => &uid_l,
        "unique_id_r" => &uid_r,
        "name_l" => &name_l,
        "name_r" => &name_r,
    )
    .unwrap()
}

fn bench_cv_jw_high_dedup(c: &mut Criterion) {
    let df = make_blocked_pairs(50_000, 30);
    let comp = ComparisonBuilder::new("name")
        .null_level()
        .exact_match_level()
        .jaro_winkler_level(0.88)
        .else_level()
        .build();
    let gamma_expr = comp.gamma_expr("gamma_").unwrap();

    c.bench_function("cv_jw_high_dedup_50k", |b| {
        b.iter(|| {
            df.clone()
                .lazy()
                .with_column(gamma_expr.clone())
                .collect()
                .unwrap()
        })
    });
}

fn bench_cv_jw_low_dedup(c: &mut Criterion) {
    let df = make_blocked_pairs(50_000, 50_000);
    let comp = ComparisonBuilder::new("name")
        .null_level()
        .exact_match_level()
        .jaro_winkler_level(0.88)
        .else_level()
        .build();
    let gamma_expr = comp.gamma_expr("gamma_").unwrap();

    c.bench_function("cv_jw_low_dedup_50k", |b| {
        b.iter(|| {
            df.clone()
                .lazy()
                .with_column(gamma_expr.clone())
                .collect()
                .unwrap()
        })
    });
}

fn bench_cv_lev_high_dedup(c: &mut Criterion) {
    let df = make_blocked_pairs(50_000, 30);
    let comp = ComparisonBuilder::new("name")
        .null_level()
        .exact_match_level()
        .levenshtein_level(2)
        .else_level()
        .build();
    let gamma_expr = comp.gamma_expr("gamma_").unwrap();

    c.bench_function("cv_lev_high_dedup_50k", |b| {
        b.iter(|| {
            df.clone()
                .lazy()
                .with_column(gamma_expr.clone())
                .collect()
                .unwrap()
        })
    });
}

fn bench_cv_lev_low_dedup(c: &mut Criterion) {
    let df = make_blocked_pairs(50_000, 50_000);
    let comp = ComparisonBuilder::new("name")
        .null_level()
        .exact_match_level()
        .levenshtein_level(2)
        .else_level()
        .build();
    let gamma_expr = comp.gamma_expr("gamma_").unwrap();

    c.bench_function("cv_lev_low_dedup_50k", |b| {
        b.iter(|| {
            df.clone()
                .lazy()
                .with_column(gamma_expr.clone())
                .collect()
                .unwrap()
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_cv_jw_high_dedup, bench_cv_jw_low_dedup, bench_cv_lev_high_dedup, bench_cv_lev_low_dedup
}
criterion_main!(benches);
