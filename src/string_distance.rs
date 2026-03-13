//! Optimised string-distance functions.
//!
//! These replace direct calls to `strsim` on the hot path with:
//!
//! - **Bounded Levenshtein**: diagonal-band DP with early termination — skips
//!   ~80% of computation when strings differ by more than the threshold.
//! - **Fast-path Jaro / Jaro-Winkler**: stack-allocated matching flags for
//!   ASCII strings under 128 bytes, eliminating heap allocation for typical
//!   person-name data.
//! - **SIMD Levenshtein** (behind the `simd` feature): delegates to
//!   `triple_accel` for hardware-accelerated edit distance.
//!
//! # Examples
//!
//! ```
//! use weldrs::string_distance::*;
//!
//! // Bounded Levenshtein — returns bool, not the actual distance
//! assert!(levenshtein_within("kitten", "sitten", 1));
//! assert!(!levenshtein_within("kitten", "sitting", 1));
//!
//! // Jaro similarity (0.0 – 1.0)
//! let j = jaro_similarity("martha", "marhta");
//! assert!(j > 0.94);
//!
//! // Jaro-Winkler similarity (adds a prefix bonus)
//! let jw = jaro_winkler_similarity("martha", "marhta");
//! assert!(jw > j);
//! ```

/// Check whether the Levenshtein distance between `a` and `b` is at most
/// `max_dist`, using a diagonal-band DP that terminates early when the
/// minimum possible distance exceeds the threshold.
///
/// # Examples
///
/// ```
/// use weldrs::string_distance::levenshtein_within;
///
/// assert!(levenshtein_within("kitten", "sitten", 1));  // distance = 1
/// assert!(!levenshtein_within("kitten", "sitting", 2)); // distance = 3
/// assert!(levenshtein_within("", "", 0));
/// ```
pub fn levenshtein_within(a: &str, b: &str, max_dist: u32) -> bool {
    #[cfg(feature = "simd")]
    {
        // triple_accel works on byte slices and returns u32 cost.
        let dist = triple_accel::levenshtein_exp(a.as_bytes(), b.as_bytes());
        dist <= max_dist
    }

    #[cfg(not(feature = "simd"))]
    {
        levenshtein_within_scalar(a, b, max_dist)
    }
}

/// Stack-allocated bounded Levenshtein for ASCII strings under 128 bytes.
///
/// Uses fixed-size `[u32; 129]` arrays on the stack instead of heap-allocated
/// `Vec<u32>`, matching the pattern used by `jaro_ascii_fast()`.
#[cfg(not(feature = "simd"))]
fn levenshtein_within_ascii_fast(a: &[u8], b: &[u8], max_dist: u32) -> bool {
    let m = a.len();
    let n = b.len();
    let max_d = max_dist as usize;

    let mut prev = [u32::MAX; 129];
    let mut curr = [u32::MAX; 129];

    // Initialise row 0.
    for (j, val) in prev[..=max_d.min(n)].iter_mut().enumerate() {
        *val = j as u32;
    }

    for i in 1..=m {
        let j_lo = i.saturating_sub(max_d);
        let j_hi = (i + max_d).min(n);

        for val in curr[j_lo..=j_hi].iter_mut() {
            *val = u32::MAX;
        }

        if j_lo == 0 {
            curr[0] = i as u32;
        }

        for j in j_lo.max(1)..=j_hi {
            let cost = if a[i - 1] == b[j - 1] { 0u32 } else { 1u32 };
            let mut val = u32::MAX;
            if prev[j] != u32::MAX {
                val = val.min(prev[j] + 1);
            }
            if curr[j - 1] != u32::MAX {
                val = val.min(curr[j - 1] + 1);
            }
            if prev[j - 1] != u32::MAX {
                val = val.min(prev[j - 1] + cost);
            }
            curr[j] = val;
        }

        let band_min = curr[j_lo..=j_hi].iter().copied().min().unwrap_or(u32::MAX);
        if band_min > max_dist {
            return false;
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n] <= max_dist
}

/// Scalar bounded Levenshtein using diagonal-band DP.
#[cfg(not(feature = "simd"))]
fn levenshtein_within_scalar(a: &str, b: &str, max_dist: u32) -> bool {
    // Fast-path: both ASCII and under 128 bytes → stack-allocated DP rows.
    if a.is_ascii() && b.is_ascii() && a.len() <= 128 && b.len() <= 128 {
        if a.len().abs_diff(b.len()) > max_dist as usize {
            return false;
        }
        return levenshtein_within_ascii_fast(a.as_bytes(), b.as_bytes(), max_dist);
    }

    // Stack-allocated char buffers for strings up to 128 chars, avoiding
    // heap allocation for typical name data. Falls back to Vec for longer strings.
    let mut a_buf = ['\0'; 128];
    let mut b_buf = ['\0'; 128];
    let mut a_len = 0usize;
    let mut b_len = 0usize;
    for ch in a.chars() {
        if a_len < 128 {
            a_buf[a_len] = ch;
        }
        a_len += 1;
    }
    for ch in b.chars() {
        if b_len < 128 {
            b_buf[b_len] = ch;
        }
        b_len += 1;
    }

    let (a_chars_vec, b_chars_vec);
    let a_chars: &[char] = if a_len <= 128 {
        &a_buf[..a_len]
    } else {
        a_chars_vec = a.chars().collect::<Vec<char>>();
        &a_chars_vec
    };
    let b_chars: &[char] = if b_len <= 128 {
        &b_buf[..b_len]
    } else {
        b_chars_vec = b.chars().collect::<Vec<char>>();
        &b_chars_vec
    };

    let m = a_chars.len();
    let n = b_chars.len();

    // Quick length check: if lengths differ by more than max_dist, impossible.
    if m.abs_diff(n) > max_dist as usize {
        return false;
    }

    let max_d = max_dist as usize;

    // Use a single-row DP with band optimisation.
    // prev[j] holds the cost for (i-1, j), curr[j] for (i, j).
    // We only need columns j in [max(0, i - max_d) .. min(n, i + max_d)].
    let width = n + 1;
    let mut prev = vec![u32::MAX; width];
    let mut curr = vec![u32::MAX; width];

    // Initialise row 0.
    for (j, val) in prev[..=max_d.min(n)].iter_mut().enumerate() {
        *val = j as u32;
    }

    for i in 1..=m {
        let j_lo = i.saturating_sub(max_d);
        let j_hi = (i + max_d).min(n);

        // Reset curr row to MAX.
        for val in curr[j_lo..=j_hi].iter_mut() {
            *val = u32::MAX;
        }

        if j_lo == 0 {
            curr[0] = i as u32;
        }

        for j in j_lo.max(1)..=j_hi {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0u32
            } else {
                1u32
            };
            let mut val = u32::MAX;
            if prev[j] != u32::MAX {
                val = val.min(prev[j] + 1); // deletion
            }
            if curr[j - 1] != u32::MAX {
                val = val.min(curr[j - 1] + 1); // insertion
            }
            if prev[j - 1] != u32::MAX {
                val = val.min(prev[j - 1] + cost); // substitution
            }
            curr[j] = val;
        }

        // Early termination: if the minimum value in the current band exceeds
        // max_dist, the final distance will also exceed it.
        let band_min = curr[j_lo..=j_hi].iter().copied().min().unwrap_or(u32::MAX);
        if band_min > max_dist {
            return false;
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n] <= max_dist
}

/// Jaro similarity with stack-allocated flags for short ASCII strings.
///
/// Returns a value between 0.0 (no similarity) and 1.0 (identical).
///
/// # Examples
///
/// ```
/// use weldrs::string_distance::jaro_similarity;
///
/// assert!((jaro_similarity("martha", "marhta") - 0.9444).abs() < 0.001);
/// assert_eq!(jaro_similarity("hello", "hello"), 1.0);
/// assert_eq!(jaro_similarity("", ""), 1.0);
/// ```
pub fn jaro_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }

    // Fast-path: both ASCII and under 128 bytes → stack-allocated flags.
    if a.is_ascii() && b.is_ascii() && a.len() <= 128 && b.len() <= 128 {
        return jaro_ascii_fast(a.as_bytes(), b.as_bytes());
    }

    // Fallback for non-ASCII or long strings.
    strsim::jaro(a, b)
}

/// Jaro-Winkler similarity with stack-allocated flags for short ASCII strings.
///
/// Extends Jaro similarity with a prefix bonus (up to 4 characters,
/// weight 0.1). Returns a value between 0.0 and 1.0.
///
/// # Examples
///
/// ```
/// use weldrs::string_distance::jaro_winkler_similarity;
///
/// let jw = jaro_winkler_similarity("martha", "marhta");
/// assert!(jw > 0.96);
/// assert_eq!(jaro_winkler_similarity("hello", "hello"), 1.0);
/// ```
pub fn jaro_winkler_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }

    // Fast-path: both ASCII and under 128 bytes → stack-allocated flags.
    if a.is_ascii() && b.is_ascii() && a.len() <= 128 && b.len() <= 128 {
        let jaro = jaro_ascii_fast(a.as_bytes(), b.as_bytes());
        // Winkler prefix bonus (up to 4 characters, weight 0.1).
        let prefix_len = a
            .bytes()
            .zip(b.bytes())
            .take(4)
            .take_while(|(x, y)| x == y)
            .count();
        return jaro + (prefix_len as f64 * 0.1 * (1.0 - jaro));
    }

    // Fallback for non-ASCII or long strings.
    strsim::jaro_winkler(a, b)
}

/// Core Jaro computation for ASCII byte slices using stack-allocated arrays.
fn jaro_ascii_fast(a: &[u8], b: &[u8]) -> f64 {
    let a_len = a.len();
    let b_len = b.len();

    let match_distance = (a_len.max(b_len) / 2).saturating_sub(1);

    let mut a_matched = [false; 128];
    let mut b_matched = [false; 128];
    let mut matches = 0u32;
    let mut transpositions = 0u32;

    // Find matches.
    for i in 0..a_len {
        let lo = i.saturating_sub(match_distance);
        let hi = (i + match_distance + 1).min(b_len);
        for j in lo..hi {
            if !b_matched[j] && a[i] == b[j] {
                a_matched[i] = true;
                b_matched[j] = true;
                matches += 1;
                break;
            }
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions.
    let mut k = 0usize;
    for i in 0..a_len {
        if a_matched[i] {
            while !b_matched[k] {
                k += 1;
            }
            if a[i] != b[k] {
                transpositions += 1;
            }
            k += 1;
        }
    }

    let m = matches as f64;
    (m / a_len as f64 + m / b_len as f64 + (m - transpositions as f64 / 2.0) / m) / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Levenshtein within ────────────────────────────────────────────

    #[test]
    fn test_levenshtein_within_exact() {
        assert!(levenshtein_within("kitten", "kitten", 0));
    }

    #[test]
    fn test_levenshtein_within_one_edit() {
        assert!(levenshtein_within("kitten", "sitten", 1));
        assert!(!levenshtein_within("kitten", "sitten", 0));
    }

    #[test]
    fn test_levenshtein_within_large_threshold() {
        assert!(levenshtein_within("abc", "xyz", 3));
    }

    #[test]
    fn test_levenshtein_within_empty() {
        assert!(levenshtein_within("", "", 0));
        assert!(levenshtein_within("abc", "", 3));
        assert!(!levenshtein_within("abc", "", 2));
    }

    #[test]
    fn test_levenshtein_within_length_filter() {
        // Strings differ by 5 chars in length, threshold=2 → early false.
        assert!(!levenshtein_within("ab", "abcdefg", 2));
    }

    #[test]
    fn test_levenshtein_within_matches_strsim() {
        let pairs = vec![
            ("kitten", "sitting"),
            ("saturday", "sunday"),
            ("", "hello"),
            ("hello", ""),
            ("abc", "abc"),
            ("abc", "axc"),
            ("flaw", "lawn"),
            ("gumbo", "gambol"),
        ];
        for (a, b) in &pairs {
            for threshold in 0..=10 {
                let expected = strsim::levenshtein(a, b) as u32 <= threshold;
                let got = levenshtein_within(a, b, threshold);
                assert_eq!(
                    got, expected,
                    "levenshtein_within({a:?}, {b:?}, {threshold}) = {got}, expected {expected}"
                );
            }
        }
    }

    // ── Jaro ──────────────────────────────────────────────────────────

    #[test]
    fn test_jaro_matches_strsim() {
        let pairs = vec![
            ("martha", "marhta"),
            ("dwayne", "duane"),
            ("dixon", "dicksonx"),
            ("abc", "xyz"),
            ("", ""),
            ("hello", "hello"),
            ("a", "b"),
        ];
        for (a, b) in &pairs {
            let expected = strsim::jaro(a, b);
            let got = jaro_similarity(a, b);
            assert!(
                (got - expected).abs() < 1e-10,
                "jaro({a:?}, {b:?}): got {got}, expected {expected}"
            );
        }
    }

    // ── Jaro-Winkler ─────────────────────────────────────────────────

    #[test]
    fn test_jaro_winkler_matches_strsim() {
        let pairs = vec![
            ("martha", "marhta"),
            ("dwayne", "duane"),
            ("dixon", "dicksonx"),
            ("abc", "xyz"),
            ("", ""),
            ("hello", "hello"),
            ("a", "b"),
        ];
        for (a, b) in &pairs {
            let expected = strsim::jaro_winkler(a, b);
            let got = jaro_winkler_similarity(a, b);
            assert!(
                (got - expected).abs() < 1e-10,
                "jaro_winkler({a:?}, {b:?}): got {got}, expected {expected}"
            );
        }
    }

    // ── Property tests: random inputs match strsim ────────────────────

    #[test]
    fn test_levenshtein_within_random_inputs() {
        // Pseudo-random test: check many pairs against strsim.
        let words: Vec<String> = (0..200).map(|i| format!("word_{i:04x}")).collect();
        for (i, a) in words.iter().enumerate() {
            for b in words.iter().skip(i).take(20) {
                for threshold in [0, 1, 2, 3, 5] {
                    let expected = strsim::levenshtein(a, b) as u32 <= threshold;
                    let got = levenshtein_within(a, b, threshold);
                    assert_eq!(got, expected, "({a}, {b}, {threshold})");
                }
            }
        }
    }

    #[test]
    fn test_jaro_winkler_random_inputs() {
        let words: Vec<String> = (0..200).map(|i| format!("word_{i:04x}")).collect();
        for (i, a) in words.iter().enumerate() {
            for b in words.iter().skip(i).take(20) {
                let expected = strsim::jaro_winkler(a, b);
                let got = jaro_winkler_similarity(a, b);
                assert!(
                    (got - expected).abs() < 1e-10,
                    "jaro_winkler({a}, {b}): got {got}, expected {expected}"
                );
            }
        }
    }
}
