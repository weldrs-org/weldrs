use criterion::{Criterion, black_box, criterion_group, criterion_main};

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

fn bench_levenshtein_short_ascii(c: &mut Criterion) {
    let names = [
        "James", "Jamxs", "Robert", "Robxrt", "Michael", "Michaxl", "William", "Willixm",
        "David", "Davxd", "Richard", "Richxrd", "Joseph", "Josxph", "Thomas", "Thomxs",
        "Charles", "Charlxs", "Daniel", "Danixl",
    ];
    let n = 100_000;
    let left: Vec<&str> = (0..n).map(|i| names[i % 10 * 2]).collect();
    let right: Vec<&str> = (0..n).map(|i| names[i % 10 * 2 + 1]).collect();

    c.bench_function("levenshtein_short_ascii_100k", |b| {
        b.iter(|| {
            for (l, r) in left.iter().zip(right.iter()) {
                black_box(weldrs::string_distance::levenshtein_within(l, r, 2));
            }
        })
    });
}

fn bench_levenshtein_unicode(c: &mut Criterion) {
    let pairs: &[(&str, &str)] = &[
        ("Jos\u{00e9}", "Jos\u{00e8}"),
        ("Fran\u{00e7}ois", "Fran\u{00e7}oix"),
        ("Ren\u{00e9}e", "Ren\u{00e8}e"),
        ("Andr\u{00e9}", "Andr\u{00e8}"),
        ("Th\u{00e9}r\u{00e8}se", "Th\u{00e9}r\u{00e9}se"),
        ("Zo\u{00eb}", "Zo\u{00e9}"),
        ("L\u{00e9}on", "L\u{00e8}on"),
        ("H\u{00e9}l\u{00e8}ne", "H\u{00e9}l\u{00e9}ne"),
        ("C\u{00e9}line", "C\u{00e8}line"),
        ("No\u{00eb}l", "No\u{00e9}l"),
    ];
    let n = 100_000;
    let left: Vec<&str> = (0..n).map(|i| pairs[i % pairs.len()].0).collect();
    let right: Vec<&str> = (0..n).map(|i| pairs[i % pairs.len()].1).collect();

    c.bench_function("levenshtein_unicode_100k", |b| {
        b.iter(|| {
            for (l, r) in left.iter().zip(right.iter()) {
                black_box(weldrs::string_distance::levenshtein_within(l, r, 2));
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
    bench_levenshtein_short_ascii,
    bench_levenshtein_unicode,
);
criterion_main!(benches);
