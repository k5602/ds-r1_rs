//! # Sampling Strategies
//!
//! Different sampling methods for text generation.

use crate::utils::error::{ModelError, Result};
use rand::Rng;

/// Sampling configuration
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
        }
    }
}

/// Text sampler for generation
pub struct Sampler {
    config: SamplingConfig,
}

impl Sampler {
    /// Create a new sampler
    pub fn new(config: SamplingConfig) -> Self {
        Self { config }
    }

    /// Sample next token using greedy decoding
    pub fn sample_greedy(&self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() {
            return Err(ModelError::Forward("Empty logits for sampling".to_string()));
        }

        // Find the token with highest probability
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| ModelError::Forward("Failed to find max logit".to_string()))?;

        Ok(max_idx as u32)
    }

    /// Sample next token using temperature sampling
    pub fn sample_temperature(&self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() {
            return Err(ModelError::Forward("Empty logits for sampling".to_string()));
        }

        let mut rng = rand::rng();
        
        // Apply temperature scaling
        let scaled_logits: Vec<f32> = if self.config.temperature > 0.0 {
            logits.iter().map(|&x| x / self.config.temperature).collect()
        } else {
            // Temperature 0 means greedy
            return self.sample_greedy(logits);
        };

        // Convert to probabilities using softmax
        let probs = self.softmax(&scaled_logits)?;

        // Sample from the distribution
        let sample: f32 = rng.random::<f32>();
        let mut cumulative = 0.0;

        for (idx, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if sample <= cumulative {
                return Ok(idx as u32);
            }
        }

        // Fallback to last token if rounding errors occur
        Ok((probs.len() - 1) as u32)
    }

    /// Sample next token using top-k sampling
    pub fn sample_top_k(&self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() {
            return Err(ModelError::Forward("Empty logits for sampling".to_string()));
        }

        let k = self.config.top_k.unwrap_or(logits.len());
        if k == 0 {
            return Err(ModelError::Forward("Top-k must be greater than 0".to_string()));
        }

        // Get top-k indices and their logits
        let mut indexed_logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(idx, &logit)| (idx, logit))
            .collect();

        // Sort by logit value (descending)
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        indexed_logits.truncate(k);

        // Create filtered logits array
        let mut filtered_logits = vec![f32::NEG_INFINITY; logits.len()];
        for (idx, logit) in indexed_logits {
            filtered_logits[idx] = logit;
        }

        // Sample from filtered distribution
        self.sample_temperature(&filtered_logits)
    }

    /// Apply softmax to convert logits to probabilities
    fn softmax(&self, logits: &[f32]) -> Result<Vec<f32>> {
        if logits.is_empty() {
            return Ok(vec![]);
        }

        // Find max for numerical stability
        let max_logit = logits
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(x - max) for each logit
        let exp_logits: Vec<f32> = logits
            .iter()
            .map(|&x| (x - max_logit).exp())
            .collect();

        // Compute sum of exponentials
        let sum_exp: f32 = exp_logits.iter().sum();

        if sum_exp <= 0.0 {
            return Err(ModelError::Forward("Invalid softmax computation".to_string()));
        }

        // Normalize to get probabilities
        let probs: Vec<f32> = exp_logits
            .iter()
            .map(|&x| x / sum_exp)
            .collect();

        Ok(probs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.repetition_penalty, 1.0);
        assert!(config.top_k.is_none());
        assert!(config.top_p.is_none());
    }

    #[test]
    fn test_sampler_creation() {
        let config = SamplingConfig::default();
        let _sampler = Sampler::new(config);
    }

    #[test]
    fn test_greedy_sampling() {
        let config = SamplingConfig::default();
        let sampler = Sampler::new(config);
        
        let logits = vec![0.1, 0.8, 0.3, 0.2];
        let token = sampler.sample_greedy(&logits).unwrap();
        assert_eq!(token, 1); // Index of highest logit (0.8)
    }

    #[test]
    fn test_temperature_sampling() {
        let mut config = SamplingConfig::default();
        config.temperature = 0.0; // Should behave like greedy
        let sampler = Sampler::new(config);
        
        let logits = vec![0.1, 0.8, 0.3, 0.2];
        let token = sampler.sample_temperature(&logits).unwrap();
        assert_eq!(token, 1); // Should be greedy when temperature is 0
    }

    #[test]
    fn test_top_k_sampling() {
        let mut config = SamplingConfig::default();
        config.top_k = Some(2);
        config.temperature = 0.0; // Make it deterministic
        let sampler = Sampler::new(config);
        
        let logits = vec![0.1, 0.8, 0.3, 0.2];
        let token = sampler.sample_top_k(&logits).unwrap();
        assert_eq!(token, 1); // Should pick from top-2: [0.8, 0.3], greedy picks 0.8
    }

    #[test]
    fn test_softmax() {
        let config = SamplingConfig::default();
        let sampler = Sampler::new(config);
        
        let logits = vec![1.0, 2.0, 3.0];
        let probs = sampler.softmax(&logits).unwrap();
        
        // Check probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check probabilities are positive
        for prob in probs {
            assert!(prob > 0.0);
        }
    }

    #[test]
    fn test_empty_logits() {
        let config = SamplingConfig::default();
        let sampler = Sampler::new(config);
        
        let logits = vec![];
        assert!(sampler.sample_greedy(&logits).is_err());
        assert!(sampler.sample_temperature(&logits).is_err());
        assert!(sampler.sample_top_k(&logits).is_err());
    }
}
