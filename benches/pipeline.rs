use criterion::{criterion_group, criterion_main, Criterion};
use polars::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use weldrs::comparison::ComparisonBuilder;
use weldrs::prelude::*;

const FIRST_NAMES: &[&str] = &[
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen",
];

const SURNAMES: &[&str] = &[
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
];

const CITIES: &[&str] = &[
    "London", "Manchester", "Birmingham", "Leeds", "Glasgow", "Liverpool",
    "Bristol", "Sheffield", "Edinburgh", "Cardiff",
];

fn generate_dataset(n: usize) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(42);
    let ids: Vec<i64> = (1..=n as i64).collect();
    let first_names: Vec<&str> = (0..n)
        .map(|_| FIRST_NAMES[rng.gen_range(0..FIRST_NAMES.len())])
        .collect();
    let surnames: Vec<&str> = (0..n)
        .map(|_| SURNAMES[rng.gen_range(0..SURNAMES.len())])
        .collect();
    let cities: Vec<&str> = (0..n)
        .map(|_| CITIES[rng.gen_range(0..CITIES.len())])
        .collect();

    df!(
        "unique_id" => &ids,
        "first_name" => &first_names,
        "surname" => &surnames,
        "city" => &cities,
    )
    .unwrap()
}

fn bench_pipeline_10k(c: &mut Criterion) {
    let df = generate_dataset(10_000);
    let lf = df.lazy();

    c.bench_function("pipeline_10k", |b| {
        b.iter(|| {
            let settings = Settings::builder(LinkType::DedupeOnly)
                .comparison(
                    ComparisonBuilder::new("first_name")
                        .null_level()
                        .exact_match_level()
                        .jaro_winkler_level(0.88)
                        .else_level()
                        .build(),
                )
                .comparison(
                    ComparisonBuilder::new("surname")
                        .null_level()
                        .exact_match_level()
                        .else_level()
                        .build(),
                )
                .blocking_rule(BlockingRule::on(&["surname"]))
                .build()
                .unwrap();

            let mut linker = Linker::new(settings).unwrap();
            linker
                .estimate_probability_two_random_records_match(
                    &lf,
                    &[BlockingRule::on(&["first_name", "surname"])],
                    1.0,
                )
                .unwrap();
            linker.estimate_u_using_random_sampling(&lf, 500).unwrap();
            linker
                .estimate_parameters_using_em(&lf, &BlockingRule::on(&["surname"]))
                .unwrap();
            let predictions = linker.predict(&lf, Some(0.0)).unwrap().collect().unwrap();
            let _clusters = linker.cluster_pairwise_predictions(&predictions, 0.5).unwrap();
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_pipeline_10k
}
criterion_main!(benches);
