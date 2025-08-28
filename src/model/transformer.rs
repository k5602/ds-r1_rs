//! # Transformer Model
//!
//! Main DeepSeek R1 model implementation with transformer layers.

use crate::model::attention::Linear;
use crate::model::config::ModelConfig;
use crate::model::embeddings::TokenEmbedding;
use crate::model::layers::{LayerNorm, TransformerLayer};
use crate::utils::error::{ModelError, Result};

/// Main DeepSeek R1 model structure
pub struct DeepSeekR1Model {
    config: ModelConfig,
    token_embedding: TokenEmbedding,
    layers: Vec<TransformerLayer>,
    final_norm: LayerNorm,
    lm_head: Linear,
}

impl DeepSeekR1Model {
    /// Create a new model with the given configuration
    pub fn new(config: ModelConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        // Initialize embeddings
        let token_embedding = TokenEmbedding::new(config.vocab_size, config.hidden_size)?;

        // Build transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let layer = TransformerLayer::new(
                config.hidden_size,
                config.num_heads,
                config.intermediate_size,
                config.max_seq_len,
                config.rope_theta,
                config.layer_norm_eps,
                config.dropout_prob,
            )?;
            layers.push(layer);
        }

        // Final normalization and LM head
        let final_norm = LayerNorm::new(config.hidden_size, config.layer_norm_eps)?;
        let lm_head = Linear::new(config.hidden_size, config.vocab_size)?;

        Ok(Self {
            config,
            token_embedding,
            layers,
            final_norm,
            lm_head,
        })
    }

    /// Get the model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Forward pass through the model
    /// Returns flattened logits with shape [seq_len * vocab_size]
    pub fn forward(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            return Err(ModelError::Forward(
                "Input token IDs cannot be empty".to_string(),
            ));
        }
        if input_ids.len() > self.config.max_seq_len {
            return Err(ModelError::Forward(format!(
                "Input sequence length {} exceeds max_seq_len {}",
                input_ids.len(),
                self.config.max_seq_len
            )));
        }

        // 1) Token embeddings
        let mut hidden = self.token_embedding.forward(input_ids)?;

        // 2) Transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }

        // 3) Final layer norm
        let hidden = self.final_norm.forward_batch(&hidden)?;

        // 4) LM head projection, flatten per-token logits
        let mut logits = Vec::with_capacity(hidden.len() * self.config.vocab_size);
        for h in &hidden {
            let token_logits = self.lm_head.forward(h)?;
            logits.extend(token_logits);
        }

        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_forward_shape() {
        let config = ModelConfig::default();
        let mut model = DeepSeekR1Model::new(config.clone()).unwrap();
        let input_ids = vec![0u32, 1u32, 2u32];

        let logits = model.forward(&input_ids).unwrap();
        assert_eq!(logits.len(), input_ids.len() * config.vocab_size);
    }

    #[test]
    fn test_forward_seq_len_validation() {
        let mut config = ModelConfig::default();
        config.max_seq_len = 2;
        let mut model = DeepSeekR1Model::new(config.clone()).unwrap();

        let input_ids = vec![0u32, 1u32, 2u32]; // length 3 > max_seq_len 2
        let result = model.forward(&input_ids);
        assert!(result.is_err());
    }
}
