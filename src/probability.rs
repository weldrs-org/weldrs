//! Probability, Bayes factor, and match-weight conversions.
//!
//! These utility functions convert between the three representations used
//! throughout the Fellegi-Sunter model: raw probabilities, Bayes factors
//! (likelihood ratios), and match weights (log2 of Bayes factors).

/// Convert a probability to a Bayes factor: p / (1 - p).
pub fn prob_to_bayes_factor(prob: f64) -> f64 {
    if prob == 1.0 {
        f64::INFINITY
    } else {
        prob / (1.0 - prob)
    }
}

/// Convert a Bayes factor back to a probability: bf / (1 + bf).
pub fn bayes_factor_to_prob(bf: f64) -> f64 {
    bf / (1.0 + bf)
}

/// Convert a probability to a match weight (log2 of the Bayes factor).
pub fn prob_to_match_weight(prob: f64) -> f64 {
    prob_to_bayes_factor(prob).log2()
}

/// Convert a match weight to a Bayes factor: 2^weight.
pub fn match_weight_to_bayes_factor(weight: f64) -> f64 {
    (2.0_f64).powf(weight)
}

/// Linearly interpolate `num_elements` values from `start` to `end` (inclusive).
fn interpolate(start: f64, end: f64, num_elements: usize) -> Vec<f64> {
    let steps = (num_elements - 1) as f64;
    let step = (end - start) / steps;
    (0..num_elements)
        .map(|i| start + (i as f64) * step)
        .collect()
}

/// Default m-probability values for `num_levels` non-null levels.
/// Highest level gets 0.95, the rest share 0.05 equally.
pub fn default_m_values(num_levels: usize) -> Vec<f64> {
    let proportion_exact_match = 0.95;
    let remainder = 1.0 - proportion_exact_match;
    let split_remainder = remainder / (num_levels - 1) as f64;
    let mut vals = vec![split_remainder; num_levels - 1];
    vals.push(proportion_exact_match);
    vals
}

/// Default u-probability values for `num_levels` non-null levels.
/// Derived from default m-values and interpolated match weights.
pub fn default_u_values(num_levels: usize) -> Vec<f64> {
    let m_vals = default_m_values(num_levels);
    let match_weights = if num_levels == 2 {
        vec![-5.0, 10.0]
    } else {
        let mut mw = interpolate(-5.0, 3.0, num_levels - 1);
        mw.push(10.0);
        mw
    };

    m_vals
        .iter()
        .zip(match_weights.iter())
        .map(|(m, w)| {
            let bf = match_weight_to_bayes_factor(*w);
            m / bf
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prob_bayes_factor_roundtrip() {
        let prob = 0.7;
        let bf = prob_to_bayes_factor(prob);
        let back = bayes_factor_to_prob(bf);
        assert!((prob - back).abs() < 1e-10);
    }

    #[test]
    fn test_prob_to_bayes_factor_one() {
        assert!(prob_to_bayes_factor(1.0).is_infinite());
    }

    #[test]
    fn test_prob_to_bayes_factor_zero() {
        assert_eq!(prob_to_bayes_factor(0.0), 0.0);
    }

    #[test]
    fn test_match_weight_roundtrip() {
        let prob = 0.8;
        let weight = prob_to_match_weight(prob);
        let bf = match_weight_to_bayes_factor(weight);
        let back = bayes_factor_to_prob(bf);
        assert!((prob - back).abs() < 1e-10);
    }

    #[test]
    fn test_default_m_values() {
        let m = default_m_values(3);
        assert_eq!(m.len(), 3);
        assert!((m.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!((m[2] - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_default_u_values() {
        let u = default_u_values(3);
        assert_eq!(u.len(), 3);
        // u-values should be small for the highest level
        assert!(u[2] < u[0]);
    }
}
