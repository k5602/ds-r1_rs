//! # Transformer Model
//!
//! Main DeepSeek R1 model implementation with transformer layers.

use crate::model::config::ModelConfig;
use crate::model::layers::{LayerNorm, TransformerLayer};
use crate::utils::error::{ModelError, Result};
use rand::Rng;

/// Trainable token embedding owned by the model (exposes mutable weights)
struct TrainableEmbedding {
    weights: Vec<Vec<f32>>,
    vocab_size: usize,
    hidden_size: usize,
    dropout_prob: f32,
}

impl TrainableEmbedding {
    fn new(vocab_size: usize, hidden_size: usize) -> Result<Self> {
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

        let mut weights = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let mut row = Vec::with_capacity(hidden_size);
            for _ in 0..hidden_size {
                row.push(rng.random::<f32>() * 2.0 * std_dev - std_dev);
            }
            weights.push(row);
        }

        Ok(Self {
            weights,
            vocab_size,
            hidden_size,
            dropout_prob: 0.1,
        })
    }

    fn forward(&self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
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
            let embedding = self.weights[token_id as usize].clone();
            embeddings.push(embedding);
        }
        let scale = (self.hidden_size as f32).sqrt();
        for embedding in &mut embeddings {
            for value in embedding {
                *value *= scale;
            }
        }
        Ok(embeddings)
    }
}

/// Trainable linear layer owned by the model (exposes mutable weights/bias)
struct TrainableLinear {
    weights: Vec<Vec<f32>>, // [out_features][in_features]
    bias: Vec<f32>,         // [out_features]
    in_features: usize,
    out_features: usize,
}

impl TrainableLinear {
    fn new(in_features: usize, out_features: usize) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(ModelError::Config(
                "Features must be greater than 0".to_string(),
            ));
        }
        let mut rng = rand::rng();
        let std_dev = (1.0 / in_features as f32).sqrt();

        let mut weights = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            let mut row = Vec::with_capacity(in_features);
            for _ in 0..in_features {
                row.push(rng.random::<f32>() * 2.0 * std_dev - std_dev);
            }
            weights.push(row);
        }
        let bias: Vec<f32> = (0..out_features)
            .map(|_| rng.random::<f32>() * 2.0 * std_dev - std_dev)
            .collect();

        Ok(Self {
            weights,
            bias,
            in_features,
            out_features,
        })
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.in_features {
            return Err(ModelError::Forward(format!(
                "Input size {} doesn't match in_features {}",
                input.len(),
                self.in_features
            )));
        }
        let mut output = vec![0.0; self.out_features];
        for (i, out) in output.iter_mut().enumerate() {
            let mut sum = self.bias[i];
            for (j, &x) in input.iter().enumerate() {
                sum += self.weights[i][j] * x;
            }
            *out = sum;
        }
        Ok(output)
    }
}

/// Main DeepSeek R1 model structure
pub struct DeepSeekR1Model {
    config: ModelConfig,
    token_embedding: TrainableEmbedding,
    layers: Vec<TransformerLayer>,
    final_norm: LayerNorm,
    lm_head: TrainableLinear,
}

impl DeepSeekR1Model {
    /// Create a new model with the given configuration
    pub fn new(config: ModelConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        // Initialize embeddings
        let token_embedding = TrainableEmbedding::new(config.vocab_size, config.hidden_size)?;

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
        let lm_head = TrainableLinear::new(config.hidden_size, config.vocab_size)?;

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

    /// Forward pass that returns the final hidden states before the LM head projection
    /// Output shape: [seq_len, hidden_size]
    pub fn forward_hidden(&mut self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
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
        Ok(hidden)
    }

    /// Mutable access to a single token embedding row
    pub fn embedding_row_mut(&mut self, token_id: usize) -> Result<&mut [f32]> {
        if token_id >= self.config.vocab_size {
            return Err(ModelError::Forward(format!(
                "Token ID {} exceeds vocabulary size {}",
                token_id, self.config.vocab_size
            )));
        }
        Ok(self.token_embedding.weights[token_id].as_mut_slice())
    }

    /// Mutable access to a single LM head weight row (for a given vocabulary index)
    pub fn lm_head_row_mut(&mut self, row: usize) -> Result<&mut [f32]> {
        if row >= self.config.vocab_size {
            return Err(ModelError::Forward(format!(
                "LM head row {} exceeds vocabulary size {}",
                row, self.config.vocab_size
            )));
        }
        Ok(self.lm_head.weights[row].as_mut_slice())
    }

    /// Mutable access to the LM head bias vector
    pub fn lm_head_bias_mut(&mut self) -> &mut [f32] {
        self.lm_head.bias.as_mut_slice()
    }

    /// Iterate all trainable parameters (embeddings rows, lm_head rows and bias)
    pub fn for_each_parameter<F: FnMut(&str, &mut [f32])>(&mut self, mut f: F) {
        for (i, row) in self.token_embedding.weights.iter_mut().enumerate() {
            let name = format!("embeddings.weight[{}]", i);
            f(&name, row.as_mut_slice());
        }
        for (i, row) in self.lm_head.weights.iter_mut().enumerate() {
            let name = format!("lm_head.weight[{}]", i);
            f(&name, row.as_mut_slice());
        }
        f("lm_head.bias", self.lm_head.bias.as_mut_slice());
    }

    /// Lightweight summary of parameter names and lengths (per-slice)
    pub fn parameters_summary(&self) -> Vec<(String, usize)> {
        let mut out = Vec::new();
        for (i, row) in self.token_embedding.weights.iter().enumerate() {
            out.push((format!("embeddings.weight[{}]", i), row.len()));
        }
        for (i, row) in self.lm_head.weights.iter().enumerate() {
            out.push((format!("lm_head.weight[{}]", i), row.len()));
        }
        out.push(("lm_head.bias".to_string(), self.lm_head.bias.len()));
        out
    }

    /// Immutable view of LM head weight matrix: [vocab_size][hidden_size]
    pub fn lm_head_weights(&self) -> &Vec<Vec<f32>> {
        &self.lm_head.weights
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
