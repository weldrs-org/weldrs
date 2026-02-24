use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn generate_pairs(n: usize) -> (Vec<String>, Vec<String>) {
    let left: Vec<String> = (0..n).map(|i| format!("name_{i}")).collect();
    let right: Vec<String> = (0..n).map(|i| format!("namx_{i}")).collect();
    (left, right)
}

fn bench_jaro_winkler(c: &mut Criterion) {
    let (left, right) = generate_pairs(100_000);
    c.bench_function("jaro_winkler_100k", |b| {
        b.iter(|| {
            for (l, r) in left.iter().zip(right.iter()) {
                black_box(weldrs::string_distance::jaro_winkler_similarity(l, r));
            }
        })
    });
}

fn bench_jaro(c: &mut Criterion) {
    let (left, right) = generate_pairs(100_000);
    c.bench_function("jaro_100k", |b| {
        b.iter(|| {
            for (l, r) in left.iter().zip(right.iter()) {
                black_box(weldrs::string_distance::jaro_similarity(l, r));
            }
        })
    });
}

fn bench_levenshtein_within(c: &mut Criterion) {
    let (left, right) = generate_pairs(100_000);
    c.bench_function("levenshtein_within_100k", |b| {
        b.iter(|| {
            for (l, r) in left.iter().zip(right.iter()) {
                black_box(weldrs::string_distance::levenshtein_within(l, r, 2));
            }
        })
    });
}

fn bench_strsim_levenshtein(c: &mut Criterion) {
    let (left, right) = generate_pairs(100_000);
    c.bench_function("strsim_levenshtein_100k", |b| {
        b.iter(|| {
            for (l, r) in left.iter().zip(right.iter()) {
                black_box(strsim::levenshtein(l, r) as u32 <= 2);
            }
        })
    });
}

fn bench_strsim_jaro_winkler(c: &mut Criterion) {
    let (left, right) = generate_pairs(100_000);
    c.bench_function("strsim_jaro_winkler_100k", |b| {
        b.iter(|| {
            for (l, r) in left.iter().zip(right.iter()) {
                black_box(strsim::jaro_winkler(l, r));
            }
        })
    });
}

criterion_group!(
    benches,
    bench_jaro_winkler,
    bench_jaro,
    bench_levenshtein_within,
    bench_strsim_levenshtein,
    bench_strsim_jaro_winkler,
);
criterion_main!(benches);
