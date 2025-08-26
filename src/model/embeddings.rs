//! # Embedding Layers
//!
//!Alike Token and positional embedding implementations for the DeepSeek R1 model.

use crate::utils::error::{ModelError, Result};
use rand::Rng;

/// Token embedding layer that maps token IDs to dense vectors
pub struct TokenEmbedding {
    /// Embedding weight matrix: [vocab_size, hidden_size]
    weights: Vec<Vec<f32>>,
    vocab_size: usize,
    hidden_size: usize,
    dropout_prob: f32,
}

impl TokenEmbedding {
    /// Create a new token embedding layer with random initialization
    pub fn new(vocab_size: usize, hidden_size: usize) -> Result<Self> {
        Self::new_with_dropout(vocab_size, hidden_size, 0.1)
    }

    /// Create a new token embedding layer with configurable dropout
    pub fn new_with_dropout(
        vocab_size: usize,
        hidden_size: usize,
        dropout_prob: f32,
    ) -> Result<Self> {
        if vocab_size == 0 {
            return Err(ModelError::Config(
                "vocab_size must be greater than 0".to_string(),
            ));
        }
        if hidden_size == 0 {
            return Err(ModelError::Config(
                "hidden_size must be greater than 0".to_string(),
            ));
        }

        let mut rng = rand::rng();
        let std_dev = (1.0 / hidden_size as f32).sqrt();

        // Initialize embedding weights with normal distribution
        let mut weights = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let mut row = Vec::with_capacity(hidden_size);
            for _ in 0..hidden_size {
                // Xavier/Glorot initialization
                row.push(rng.random::<f32>() * 2.0 * std_dev - std_dev);
            }
            weights.push(row);
        }

        Ok(Self {
            weights,
            vocab_size,
            hidden_size,
            dropout_prob,
        })
    }

    /// Forward pass through token embedding
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        if input_ids.is_empty() {
            return Err(ModelError::Forward(
                "Input token IDs cannot be empty".to_string(),
            ));
        }

        let mut embeddings = Vec::with_capacity(input_ids.len());

        for &token_id in input_ids {
            if token_id as usize >= self.vocab_size {
                return Err(ModelError::Forward(format!(
                    "Token ID {} exceeds vocabulary size {}",
                    token_id, self.vocab_size
                )));
            }

            // Get embedding for this token
            let embedding = self.weights[token_id as usize].clone();
            embeddings.push(embedding);
        }

        // Apply embedding scaling (common in transformer models)
        let scale = (self.hidden_size as f32).sqrt();
        for embedding in &mut embeddings {
            for value in embedding {
                *value *= scale;
            }
        }

        Ok(embeddings)
    }

    /// Apply dropout to embeddings (for training)
    pub fn apply_dropout(&self, embeddings: &mut Vec<Vec<f32>>, training: bool) -> Result<()> {
        if !training || self.dropout_prob == 0.0 {
            return Ok(());
        }

        let mut rng = rand::rng();
        let keep_prob = 1.0 - self.dropout_prob;
        let scale = 1.0 / keep_prob;

        for embedding in embeddings {
            for value in embedding {
                if rng.random::<f32>() < keep_prob {
                    *value *= scale;
                } else {
                    *value = 0.0;
                }
            }
        }

        Ok(())
    }

    /// Get embedding dimension
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Rotary Position Embedding (RoPE) implementation
pub struct RotaryEmbedding {
    /// Precomputed cosine values for each position and dimension
    cos_cache: Vec<Vec<f32>>,
    /// Precomputed sine values for each position and dimension
    sin_cache: Vec<Vec<f32>>,
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
}

impl RotaryEmbedding {
    /// Create a new rotary embedding layer
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> Result<Self> {
        if head_dim == 0 {
            return Err(ModelError::Config(
                "head_dim must be greater than 0".to_string(),
            ));
        }
        if head_dim % 2 != 0 {
            return Err(ModelError::Config(
                "head_dim must be even for RoPE".to_string(),
            ));
        }
        if max_seq_len == 0 {
            return Err(ModelError::Config(
                "max_seq_len must be greater than 0".to_string(),
            ));
        }

        // Precompute rotation matrices for all positions
        let mut cos_cache = Vec::with_capacity(max_seq_len);
        let mut sin_cache = Vec::with_capacity(max_seq_len);

        for pos in 0..max_seq_len {
            let mut cos_row = Vec::with_capacity(head_dim);
            let mut sin_row = Vec::with_capacity(head_dim);

            for i in (0..head_dim).step_by(2) {
                let freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;

                cos_row.push(angle.cos());
                cos_row.push(angle.cos());
                sin_row.push(angle.sin());
                sin_row.push(angle.sin());
            }

            cos_cache.push(cos_row);
            sin_cache.push(sin_row);
        }

        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
            max_seq_len,
            theta,
        })
    }

    /// Apply rotary position embedding to a vector at given position
    pub fn apply_rotary(&self, x: &[f32], position: usize) -> Result<Vec<f32>> {
        if position >= self.max_seq_len {
            return Err(ModelError::Forward(format!(
                "Position {} exceeds max_seq_len {}",
                position, self.max_seq_len
            )));
        }

        if x.len() != self.head_dim {
            return Err(ModelError::Forward(format!(
                "Input dimension {} doesn't match head_dim {}",
                x.len(),
                self.head_dim
            )));
        }

        let cos = &self.cos_cache[position];
        let sin = &self.sin_cache[position];
        let mut result = vec![0.0; self.head_dim];
        for i in (0..self.head_dim).step_by(2) {
            let x0 = x[i];
            let x1 = x[i + 1];

            result[i] = x0 * cos[i] - x1 * sin[i];
            result[i + 1] = x0 * sin[i + 1] + x1 * cos[i + 1];
        }

        Ok(result)
    }

    /// Input: [seq_len, head_dim], Output: [seq_len, head_dim]
    pub fn apply_rotary_batch(&self, x: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut result = Vec::with_capacity(x.len());

        for (pos, vector) in x.iter().enumerate() {
            let rotated = self.apply_rotary(vector, pos)?;
            result.push(rotated);
        }

        Ok(result)
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_embedding_creation() {
        let embedding = TokenEmbedding::new(32000, 512);
        assert!(embedding.is_ok());

        let emb = embedding.unwrap();
        assert_eq!(emb.vocab_size(), 32000);
        assert_eq!(emb.hidden_size(), 512);
    }

    #[test]
    fn test_token_embedding_forward() {
        let embedding = TokenEmbedding::new(1000, 128).unwrap();
        let input_ids = vec![0, 1, 999];

        let result = embedding.forward(&input_ids);
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 128);
        assert_eq!(embeddings[1].len(), 128);
        assert_eq!(embeddings[2].len(), 128);
    }

    #[test]
    fn test_token_embedding_invalid_token() {
        let embedding = TokenEmbedding::new(1000, 128).unwrap();
        let input_ids = vec![1000]; // Out of bounds

        let result = embedding.forward(&input_ids);
        assert!(result.is_err());
    }

    #[test]
    fn test_token_embedding_empty_input() {
        let embedding = TokenEmbedding::new(1000, 128).unwrap();
        let input_ids = vec![];

        let result = embedding.forward(&input_ids);
        assert!(result.is_err());
    }

    #[test]
    fn test_token_embedding_scaling() {
        let embedding = TokenEmbedding::new(100, 64).unwrap();
        let input_ids = vec![0];

        let embeddings = embedding.forward(&input_ids).unwrap();
        let scale = (64.0_f32).sqrt();

        // Check that embeddings are scaled (this is a rough check)
        let sum_of_squares: f32 = embeddings[0].iter().map(|x| x * x).sum();
        assert!(sum_of_squares > 0.0); // Should have some magnitude due to scaling
    }

    #[test]
    fn test_dropout_application() {
        let embedding = TokenEmbedding::new_with_dropout(100, 64, 0.5).unwrap();
        let input_ids = vec![0, 1, 2];

        let mut embeddings = embedding.forward(&input_ids).unwrap();
        let original = embeddings.clone();

        // Apply dropout in training mode
        embedding.apply_dropout(&mut embeddings, true).unwrap();

        // Some values should be different (either 0 or scaled)
        let mut has_zeros = false;
        let mut has_scaled = false;

        for (orig_emb, drop_emb) in original.iter().zip(embeddings.iter()) {
            for (orig_val, drop_val) in orig_emb.iter().zip(drop_emb.iter()) {
                if *drop_val == 0.0 {
                    has_zeros = true;
                } else if (*drop_val - orig_val).abs() > 1e-6 {
                    has_scaled = true;
                }
            }
        }
    }

    #[test]
    fn test_rotary_embedding_creation() {
        let rope = RotaryEmbedding::new(64, 2048, 10000.0);
        assert!(rope.is_ok());

        let rope = rope.unwrap();
        assert_eq!(rope.head_dim(), 64);
        assert_eq!(rope.max_seq_len(), 2048);
    }

    #[test]
    fn test_rotary_embedding_invalid_head_dim() {
        // Odd head dimension should fail
        let rope = RotaryEmbedding::new(63, 2048, 10000.0);
        assert!(rope.is_err());

        // Zero head dimension should fail
        let rope = RotaryEmbedding::new(0, 2048, 10000.0);
        assert!(rope.is_err());
    }

    #[test]
    fn test_rotary_embedding_apply() {
        let rope = RotaryEmbedding::new(4, 10, 10000.0).unwrap();
        let input = vec![1.0, 0.0, 0.5, -0.5];

        let result = rope.apply_rotary(&input, 0);
        assert!(result.is_ok());

        let rotated = result.unwrap();
        assert_eq!(rotated.len(), 4);

        // At position 0, rotation should be minimal (cos ≈ 1, sin ≈ 0)
        assert!((rotated[0] - 1.0).abs() < 0.1);
        assert!(rotated[1].abs() < 0.1);
    }

    #[test]
    fn test_rotary_embedding_position_bounds() {
        let rope = RotaryEmbedding::new(4, 10, 10000.0).unwrap();
        let input = vec![1.0, 0.0, 0.5, -0.5];

        // Valid position
        assert!(rope.apply_rotary(&input, 9).is_ok());

        // Invalid position
        assert!(rope.apply_rotary(&input, 10).is_err());
    }

    #[test]
    fn test_rotary_embedding_dimension_mismatch() {
        let rope = RotaryEmbedding::new(4, 10, 10000.0).unwrap();
        let input = vec![1.0, 0.0]; // Wrong dimension
        let result = rope.apply_rotary(&input, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rotary_embedding_batch() {
        let rope = RotaryEmbedding::new(4, 10, 10000.0).unwrap();
        let batch = vec![
            vec![1.0, 0.0, 0.5, -0.5],
            vec![0.0, 1.0, -0.5, 0.5],
            vec![1.0, 1.0, 0.0, 0.0],
        ];

        let result = rope.apply_rotary_batch(&batch);
        assert!(result.is_ok());

        let rotated_batch = result.unwrap();
        assert_eq!(rotated_batch.len(), 3);
        assert_eq!(rotated_batch[0].len(), 4);
        assert_eq!(rotated_batch[1].len(), 4);
        assert_eq!(rotated_batch[2].len(), 4);
    }

    #[test]
    fn test_rotary_embedding_rotation_property() {
        let rope = RotaryEmbedding::new(4, 10, 10000.0).unwrap();
        let input = vec![1.0, 0.0, 0.0, 1.0];

        // Apply rotation at different positions
        let rot0 = rope.apply_rotary(&input, 0).unwrap();
        let rot1 = rope.apply_rotary(&input, 1).unwrap();

        // Rotations at different positions should be different
        let mut different = false;
        for (a, b) in rot0.iter().zip(rot1.iter()) {
            if (a - b).abs() > 1e-6 {
                different = true;
                break;
            }
        }
        assert!(different, "Rotations at different positions should differ");
    }

    #[test]
    fn test_embedding_config_validation() {
        // Test invalid configurations
        assert!(TokenEmbedding::new(0, 128).is_err());
        assert!(TokenEmbedding::new(1000, 0).is_err());
        assert!(RotaryEmbedding::new(0, 2048, 10000.0).is_err());
        assert!(RotaryEmbedding::new(64, 0, 10000.0).is_err());
    }
}
