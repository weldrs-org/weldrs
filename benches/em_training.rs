use criterion::{black_box, criterion_group, criterion_main, Criterion};
use polars::prelude::*;
use weldrs::comparison::ComparisonBuilder;
use weldrs::em::expectation_maximisation;
use weldrs::settings::TrainingSettings;

fn make_synthetic_patterns(n_patterns: usize, n_comparisons: usize) -> (LazyFrame, Vec<weldrs::comparison::Comparison>) {
    let mut columns: Vec<Column> = Vec::new();
    let mut comparisons = Vec::new();

    for c in 0..n_comparisons {
        let name = format!("col_{c}");
        // Random gamma values: -1, 0, 1, 2
        let gammas: Vec<i8> = (0..n_patterns)
            .map(|i| ((i + c) % 4) as i8 - 1)
            .collect();
        columns.push(Column::new(format!("gamma_{name}").into(), &gammas));

        comparisons.push(
            ComparisonBuilder::new(&name)
                .null_level()
                .exact_match_level()
                .jaro_winkler_level(0.88)
                .else_level()
                .build(),
        );
    }

    // Add count column
    let counts: Vec<u32> = (0..n_patterns).map(|i| (i as u32 % 100) + 1).collect();
    columns.push(Column::new("__count".into(), &counts));

    let df = DataFrame::new(columns).unwrap();
    (df.lazy(), comparisons)
}

fn bench_em_500_patterns_3_comparisons(c: &mut Criterion) {
    let (cv, comparisons) = make_synthetic_patterns(500, 3);
    let training = TrainingSettings {
        em_convergence: 0.0001,
        max_iterations: 25,
        ..Default::default()
    };

    c.bench_function("em_500pat_3comp", |b| {
        b.iter(|| {
            black_box(
                expectation_maximisation(&cv, comparisons.clone(), 0.05, &training, "gamma_", &[])
                    .unwrap(),
            )
        })
    });
}

fn bench_em_1000_patterns_5_comparisons(c: &mut Criterion) {
    let (cv, comparisons) = make_synthetic_patterns(1000, 5);
    let training = TrainingSettings {
        em_convergence: 0.0001,
        max_iterations: 25,
        ..Default::default()
    };

    c.bench_function("em_1000pat_5comp", |b| {
        b.iter(|| {
            black_box(
                expectation_maximisation(&cv, comparisons.clone(), 0.05, &training, "gamma_", &[])
                    .unwrap(),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_em_500_patterns_3_comparisons,
    bench_em_1000_patterns_5_comparisons,
);
criterion_main!(benches);
