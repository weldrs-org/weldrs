use criterion::{Criterion, criterion_group, criterion_main};
use polars::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use weldrs::comparison::{Comparison, ComparisonBuilder};
use weldrs::predict::{predict, predict_direct};

/// Build `n` trained Comparison objects with set m/u probabilities.
fn make_trained_comparisons(n: usize) -> Vec<Comparison> {
    (0..n)
        .map(|i| {
            let name = format!("col_{i}");
            let mut comp = ComparisonBuilder::new(&name)
                .null_level()
                .exact_match_level()
                .else_level()
                .build()
                .unwrap();

            for level in &mut comp.comparison_levels {
                if level.is_null_level {
                    continue;
                }
                if level.comparison_vector_value == 1 {
                    level.m_probability = Some(0.95);
                    level.u_probability = Some(0.01);
                } else {
                    level.m_probability = Some(0.05);
                    level.u_probability = Some(0.99);
                }
            }
            comp
        })
        .collect()
}

/// Build a DataFrame with gamma columns filled with random 0/1 values.
fn make_gamma_df(n_rows: usize, comparisons: &[Comparison]) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(42);

    let uid_l: Vec<i64> = (0..n_rows as i64).collect();
    let uid_r: Vec<i64> = (n_rows as i64..2 * n_rows as i64).collect();

    let mut columns: Vec<Column> = vec![
        Column::new("unique_id_l".into(), &uid_l),
        Column::new("unique_id_r".into(), &uid_r),
    ];

    for comp in comparisons {
        let col_name = comp.gamma_column_name("gamma_");
        let gammas: Vec<i8> = (0..n_rows).map(|_| rng.gen_range(0..=1i8)).collect();
        columns.push(Column::new(col_name.into(), &gammas));
    }

    DataFrame::new(n_rows, columns).unwrap()
}

fn bench_predict(c: &mut Criterion) {
    let comparisons = make_trained_comparisons(3);

    for &n in &[1_000, 10_000, 100_000] {
        let df = make_gamma_df(n, &comparisons);
        let mut group = c.benchmark_group(format!("predict_{}k", n / 1000));

        group.bench_function("lazy", |b| {
            b.iter(|| {
                predict(
                    df.clone().lazy(),
                    &comparisons,
                    0.05,
                    "gamma_",
                    "bf_",
                    None,
                    None,
                )
                .unwrap()
                .collect()
                .unwrap()
            })
        });

        group.bench_function("direct", |b| {
            b.iter(|| {
                predict_direct(df.clone(), &comparisons, 0.05, "gamma_", "bf_", None, None).unwrap()
            })
        });

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_predict
}
criterion_main!(benches);
