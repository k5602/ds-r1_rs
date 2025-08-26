//! # Mathematical Utilities
//!
//! Mathematical functions and utilities for the model.

use crate::utils::error::{ModelError, Result};

/// Mathematical utility functions
pub struct MathUtils;

impl MathUtils {
    /// Compute softmax over a vector
    pub fn softmax(input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Err(ModelError::Math("Input vector is empty".to_string()));
        }
        
        // Find maximum for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exponentials
        let exp_values: Vec<f32> = input.iter()
            .map(|&x| (x - max_val).exp())
            .collect();
        
        // Compute sum
        let sum: f32 = exp_values.iter().sum();
        
        if sum == 0.0 {
            return Err(ModelError::Math("Softmax sum is zero".to_string()));
        }
        
        // Normalize
        Ok(exp_values.iter().map(|&x| x / sum).collect())
    }
    
    /// Compute layer normalization
    pub fn layer_norm(input: &[f32], eps: f32) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Err(ModelError::Math("Input vector is empty".to_string()));
        }
        
        let n = input.len() as f32;
        
        // Compute mean
        let mean = input.iter().sum::<f32>() / n;
        
        // Compute variance
        let variance = input.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / n;
        
        // Normalize
        let std_dev = (variance + eps).sqrt();
        Ok(input.iter().map(|&x| (x - mean) / std_dev).collect())
    }
    
    /// Compute GELU activation
    pub fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }
    
    /// Compute SwiGLU activation
    pub fn swiglu(x: f32, gate: f32) -> f32 {
        x * Self::sigmoid(gate)
    }
    
    /// Compute sigmoid activation
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Compute ReLU activation
    pub fn relu(x: f32) -> f32 {
        x.max(0.0)
    }
    
    /// Generate random normal distribution sample (Box-Muller transform)
    pub fn random_normal(mean: f32, std_dev: f32) -> f32 {
        use rand::Rng;
        let mut rng = rand::rng();
        
        let u1: f32 = rng.random();
        let u2: f32 = rng.random();
        
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        mean + std_dev * z0
    }
    
    /// Compute cosine similarity between two vectors
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(ModelError::Math("Vector dimensions must match".to_string()));
        }
        
        if a.is_empty() {
            return Err(ModelError::Math("Vectors cannot be empty".to_string()));
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return Err(ModelError::Math("Cannot compute similarity with zero vector".to_string()));
        }
        
        Ok(dot_product / (norm_a * norm_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let result = MathUtils::softmax(&input).unwrap();
        
        // Check that probabilities sum to 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check that all values are positive
        assert!(result.iter().all(|&x| x > 0.0));
        
        // Check that larger inputs have larger probabilities
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_layer_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = MathUtils::layer_norm(&input, 1e-5).unwrap();
        
        // Check that mean is approximately 0
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean.abs() < 1e-6);
        
        // Check that variance is approximately 1
        let variance: f32 = result.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / result.len() as f32;
        assert!((variance - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_activations() {
        // Test GELU
        assert!(MathUtils::gelu(0.0).abs() < 1e-6);
        assert!(MathUtils::gelu(1.0) > 0.0);
        assert!(MathUtils::gelu(-1.0) < 0.0);
        
        // Test sigmoid
        assert!((MathUtils::sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(MathUtils::sigmoid(10.0) > 0.9);
        assert!(MathUtils::sigmoid(-10.0) < 0.1);
        
        // Test ReLU
        assert_eq!(MathUtils::relu(5.0), 5.0);
        assert_eq!(MathUtils::relu(-5.0), 0.0);
        assert_eq!(MathUtils::relu(0.0), 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = MathUtils::cosine_similarity(&a, &b).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6);
        
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let similarity = MathUtils::cosine_similarity(&a, &b).unwrap();
        assert!(similarity.abs() < 1e-6);
    }

    #[test]
    fn test_error_cases() {
        // Empty vector for softmax
        assert!(MathUtils::softmax(&[]).is_err());
        
        // Empty vector for layer norm
        assert!(MathUtils::layer_norm(&[], 1e-5).is_err());
        
        // Mismatched vector sizes for cosine similarity
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert!(MathUtils::cosine_similarity(&a, &b).is_err());
    }
}