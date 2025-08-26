//! # Neural Network Layers
//!
//! Basic neural network layers used throughout the model.

use crate::model::attention::Linear;
use crate::utils::error::{ModelError, Result};
use rand::Rng;

/// Transformer layer combining attention and feed-forward with residual connections
/// Uses pre-normalization architecture (LayerNorm before attention/FFN)
pub struct TransformerLayer {
    /// Self-attention mechanism
    attention: crate::model::attention::StandardAttention,
    /// Feed-forward network
    feed_forward: FeedForward,
    /// Layer normalization before attention
    attention_norm: LayerNorm,
    /// Layer normalization before feed-forward
    ffn_norm: LayerNorm,
    hidden_size: usize,
}

impl TransformerLayer {
    /// Create a new transformer layer
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        max_seq_len: usize,
        rope_theta: f32,
        layer_norm_eps: f32,
        dropout_prob: f32,
    ) -> Result<Self> {
        if hidden_size == 0 || num_heads == 0 || intermediate_size == 0 {
            return Err(ModelError::Config(
                "All sizes must be greater than 0".to_string(),
            ));
        }

        // Create attention mechanism
        let attention = crate::model::attention::StandardAttention::new(
            hidden_size,
            num_heads,
            max_seq_len,
            rope_theta,
        )?;

        // Create feed-forward network
        let feed_forward =
            FeedForward::new_with_dropout(hidden_size, intermediate_size, dropout_prob)?;

        // Create layer normalizations
        let attention_norm = LayerNorm::new(hidden_size, layer_norm_eps)?;
        let ffn_norm = LayerNorm::new(hidden_size, layer_norm_eps)?;

        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
            hidden_size,
        })
    }

    /// Create a new transformer layer with default parameters
    pub fn new_default(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        Self::new(
            hidden_size,
            num_heads,
            intermediate_size,
            2048,    // max_seq_len
            10000.0, // rope_theta
            1e-5,    // layer_norm_eps
            0.1,     // dropout_prob
        )
    }

    /// Add residual connection: output = input + residual
    fn add_residual(&self, input: &[Vec<f32>], residual: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if input.len() != residual.len() {
            return Err(ModelError::Forward(
                "Input and residual must have same sequence length".to_string(),
            ));
        }

        let mut output = Vec::with_capacity(input.len());

        for (inp, res) in input.iter().zip(residual.iter()) {
            if inp.len() != res.len() {
                return Err(ModelError::Forward(
                    "Input and residual must have same hidden size".to_string(),
                ));
            }

            let mut sum = Vec::with_capacity(inp.len());
            for (&i, &r) in inp.iter().zip(res.iter()) {
                sum.push(i + r);
            }
            output.push(sum);
        }

        Ok(output)
    }

    /// Forward pass through transformer layer with pre-normalization
    /// Input: [seq_len, hidden_size], Output: [seq_len, hidden_size]
    pub fn forward(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        self.forward_with_training(input, false)
    }

    /// Forward pass with training mode control
    pub fn forward_with_training(
        &self,
        input: &[Vec<f32>],
        training: bool,
    ) -> Result<Vec<Vec<f32>>> {
        if input.is_empty() {
            return Err(ModelError::Forward(
                "Input sequence cannot be empty".to_string(),
            ));
        }

        // Validate input dimensions
        for (i, seq) in input.iter().enumerate() {
            if seq.len() != self.hidden_size {
                return Err(ModelError::Forward(format!(
                    "Input at position {} has size {} but expected {}",
                    i,
                    seq.len(),
                    self.hidden_size
                )));
            }
        }

        // Pre-normalization + Self-attention + Residual connection
        let attention_input = self.attention_norm.forward_batch(input)?;
        let attention_output = self.attention.forward(&attention_input)?;
        let after_attention = self.add_residual(input, &attention_output)?;

        // Pre-normalization + Feed-forward + Residual connection
        let ffn_input = self.ffn_norm.forward_batch(&after_attention)?;
        let ffn_output = self
            .feed_forward
            .forward_batch_with_training(&ffn_input, training)?;
        let final_output = self.add_residual(&after_attention, &ffn_output)?;

        Ok(final_output)
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get number of attention heads
    pub fn num_heads(&self) -> usize {
        self.attention.num_heads()
    }

    /// Get intermediate size
    pub fn intermediate_size(&self) -> usize {
        self.feed_forward.intermediate_size()
    }
}

/// Feed-forward network with SwiGLU activation
/// Implements the gated linear unit variant used in modern transformers
pub struct FeedForward {
    /// Gate projection (first linear layer)
    gate_proj: Linear,
    /// Up projection (second linear layer, parallel to gate)
    up_proj: Linear,
    /// Down projection (final linear layer)
    down_proj: Linear,
    hidden_size: usize,
    intermediate_size: usize,
    dropout_prob: f32,
}

impl FeedForward {
    /// Create a new feed-forward layer with SwiGLU activation
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        Self::new_with_dropout(hidden_size, intermediate_size, 0.1)
    }

    /// Create a new feed-forward layer with configurable dropout
    pub fn new_with_dropout(
        hidden_size: usize,
        intermediate_size: usize,
        dropout_prob: f32,
    ) -> Result<Self> {
        if hidden_size == 0 || intermediate_size == 0 {
            return Err(ModelError::Config(
                "hidden_size and intermediate_size must be greater than 0".to_string(),
            ));
        }

        // Create the three linear projections
        let gate_proj = Linear::new(hidden_size, intermediate_size)?;
        let up_proj = Linear::new(hidden_size, intermediate_size)?;
        let down_proj = Linear::new(intermediate_size, hidden_size)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            hidden_size,
            intermediate_size,
            dropout_prob,
        })
    }

    /// SwiGLU activation function: swish(gate) * up
    /// where swish(x) = x * sigmoid(x)
    fn swiglu_activation(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        if gate.len() != up.len() {
            return Err(ModelError::Forward(
                "Gate and up projections must have same size".to_string(),
            ));
        }

        let mut result = Vec::with_capacity(gate.len());

        for (&g, &u) in gate.iter().zip(up.iter()) {
            // Swish activation: x * sigmoid(x)
            let swish_g = g * sigmoid(g);
            result.push(swish_g * u);
        }

        Ok(result)
    }

    /// Apply dropout to intermediate activations
    fn apply_dropout(&self, input: &mut [f32], training: bool) -> Result<()> {
        if !training || self.dropout_prob == 0.0 {
            return Ok(());
        }

        let mut rng = rand::rng();
        let keep_prob = 1.0 - self.dropout_prob;
        let scale = 1.0 / keep_prob;

        for val in input {
            if rng.random::<f32>() < keep_prob {
                *val *= scale;
            } else {
                *val = 0.0;
            }
        }

        Ok(())
    }

    /// Forward pass through feed-forward layer
    /// Input: [hidden_size], Output: [hidden_size]
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.forward_with_training(input, false)
    }

    /// Forward pass with training mode control
    pub fn forward_with_training(&self, input: &[f32], training: bool) -> Result<Vec<f32>> {
        if input.len() != self.hidden_size {
            return Err(ModelError::Forward(format!(
                "Input size {} doesn't match hidden_size {}",
                input.len(),
                self.hidden_size
            )));
        }

        // Apply gate and up projections
        let gate_output = self.gate_proj.forward(input)?;
        let up_output = self.up_proj.forward(input)?;

        // Apply SwiGLU activation
        let mut activated = self.swiglu_activation(&gate_output, &up_output)?;

        // Apply dropout to intermediate activations
        self.apply_dropout(&mut activated, training)?;

        // Apply down projection
        let output = self.down_proj.forward(&activated)?;

        Ok(output)
    }

    /// Forward pass for batch: [batch_size, hidden_size] -> [batch_size, hidden_size]
    pub fn forward_batch(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        self.forward_batch_with_training(input, false)
    }

    /// Forward pass for batch with training mode control
    pub fn forward_batch_with_training(
        &self,
        input: &[Vec<f32>],
        training: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(input.len());

        for input_vec in input {
            let output_vec = self.forward_with_training(input_vec, training)?;
            output.push(output_vec);
        }

        Ok(output)
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get intermediate size
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
}

/// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Layer normalization implementation
/// Normalizes inputs across the feature dimension
pub struct LayerNorm {
    /// Learnable scale parameters
    weight: Vec<f32>,
    /// Learnable bias parameters
    bias: Vec<f32>,
    /// Small epsilon for numerical stability
    eps: f32,
    hidden_size: usize,
}

impl LayerNorm {
    /// Create a new layer normalization layer
    pub fn new(hidden_size: usize, eps: f32) -> Result<Self> {
        if hidden_size == 0 {
            return Err(ModelError::Config(
                "hidden_size must be greater than 0".to_string(),
            ));
        }

        if eps <= 0.0 {
            return Err(ModelError::Config("eps must be greater than 0".to_string()));
        }

        // Initialize weight to 1.0 and bias to 0.0
        let weight = vec![1.0; hidden_size];
        let bias = vec![0.0; hidden_size];

        Ok(Self {
            weight,
            bias,
            eps,
            hidden_size,
        })
    }

    /// Forward pass through layer normalization
    /// Input: [hidden_size], Output: [hidden_size]
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.hidden_size {
            return Err(ModelError::Forward(format!(
                "Input size {} doesn't match hidden_size {}",
                input.len(),
                self.hidden_size
            )));
        }

        // Compute mean
        let mean = input.iter().sum::<f32>() / input.len() as f32;

        // Compute variance
        let variance = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;

        // Normalize and apply scale and shift
        let std_dev = (variance + self.eps).sqrt();
        let mut output = Vec::with_capacity(input.len());

        for (i, &x) in input.iter().enumerate() {
            let normalized = (x - mean) / std_dev;
            let scaled_shifted = normalized * self.weight[i] + self.bias[i];
            output.push(scaled_shifted);
        }

        Ok(output)
    }

    /// Forward pass for batch: [batch_size, hidden_size] -> [batch_size, hidden_size]
    pub fn forward_batch(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(input.len());

        for input_vec in input {
            let output_vec = self.forward(input_vec)?;
            output.push(output_vec);
        }

        Ok(output)
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get epsilon value
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_function() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(1000.0) > 0.99); // Should approach 1
        assert!(sigmoid(-1000.0) < 0.01); // Should approach 0

        // Test monotonicity
        assert!(sigmoid(-1.0) < sigmoid(0.0));
        assert!(sigmoid(0.0) < sigmoid(1.0));
    }

    #[test]
    fn test_feedforward_creation() {
        let ff = FeedForward::new(512, 2048);
        assert!(ff.is_ok());

        let layer = ff.unwrap();
        assert_eq!(layer.hidden_size(), 512);
        assert_eq!(layer.intermediate_size(), 2048);
    }

    #[test]
    fn test_feedforward_invalid_config() {
        assert!(FeedForward::new(0, 2048).is_err());
        assert!(FeedForward::new(512, 0).is_err());
    }

    #[test]
    fn test_feedforward_forward() {
        let ff = FeedForward::new(64, 256).unwrap();
        let input = vec![1.0; 64];

        let result = ff.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_feedforward_batch() {
        let ff = FeedForward::new(32, 128).unwrap();
        let input = vec![vec![1.0; 32], vec![0.5; 32], vec![-0.5; 32]];

        let result = ff.forward_batch(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].len(), 32);
        assert_eq!(output[1].len(), 32);
        assert_eq!(output[2].len(), 32);
    }

    #[test]
    fn test_feedforward_wrong_input_size() {
        let ff = FeedForward::new(64, 256).unwrap();
        let input = vec![1.0; 32]; // Wrong size

        let result = ff.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_swiglu_activation() {
        let ff = FeedForward::new(4, 8).unwrap();
        let gate = vec![1.0, 0.0, -1.0, 2.0];
        let up = vec![2.0, 1.0, 0.5, -1.0];

        let result = ff.swiglu_activation(&gate, &up);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 4);

        // Check that SwiGLU produces reasonable values
        // swish(1.0) * 2.0 should be positive
        assert!(output[0] > 0.0);
        // swish(0.0) * 1.0 should be 0
        assert!((output[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_swiglu_mismatched_sizes() {
        let ff = FeedForward::new(4, 8).unwrap();
        let gate = vec![1.0, 2.0];
        let up = vec![1.0, 2.0, 3.0]; // Different size

        let result = ff.swiglu_activation(&gate, &up);
        assert!(result.is_err());
    }

    #[test]
    fn test_feedforward_with_dropout() {
        let ff = FeedForward::new_with_dropout(32, 128, 0.5).unwrap();
        let input = vec![1.0; 32];

        // Test training mode (with dropout)
        let result_train = ff.forward_with_training(&input, true);
        assert!(result_train.is_ok());

        // Test inference mode (no dropout)
        let result_infer = ff.forward_with_training(&input, false);
        assert!(result_infer.is_ok());

        let output_train = result_train.unwrap();
        let output_infer = result_infer.unwrap();

        assert_eq!(output_train.len(), 32);
        assert_eq!(output_infer.len(), 32);
    }

    #[test]
    fn test_layer_norm_creation() {
        let ln = LayerNorm::new(512, 1e-5);
        assert!(ln.is_ok());

        let layer = ln.unwrap();
        assert_eq!(layer.hidden_size(), 512);
        assert_eq!(layer.eps(), 1e-5);
    }

    #[test]
    fn test_layer_norm_invalid_config() {
        assert!(LayerNorm::new(0, 1e-5).is_err());
        assert!(LayerNorm::new(512, 0.0).is_err());
        assert!(LayerNorm::new(512, -1e-5).is_err());
    }

    #[test]
    fn test_layer_norm_forward() {
        let ln = LayerNorm::new(4, 1e-5).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let result = ln.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 4);

        // Check that output has approximately zero mean and unit variance
        let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
        let variance: f32 =
            output.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / output.len() as f32;

        assert!(mean.abs() < 1e-5); // Should be close to zero
        assert!((variance - 1.0).abs() < 1e-3); // Should be close to 1
    }

    #[test]
    fn test_layer_norm_batch() {
        let ln = LayerNorm::new(3, 1e-5).unwrap();
        let input = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![-1.0, 0.0, 1.0],
        ];

        let result = ln.forward_batch(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].len(), 3);
        assert_eq!(output[1].len(), 3);
        assert_eq!(output[2].len(), 3);
    }

    #[test]
    fn test_layer_norm_wrong_input_size() {
        let ln = LayerNorm::new(4, 1e-5).unwrap();
        let input = vec![1.0, 2.0]; // Wrong size

        let result = ln.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_norm_constant_input() {
        let ln = LayerNorm::new(4, 1e-5).unwrap();
        let input = vec![5.0, 5.0, 5.0, 5.0]; // All same values

        let result = ln.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        // With constant input, output should be all zeros (after normalization)
        for &val in &output {
            assert!(val.abs() < 1e-3);
        }
    }

    #[test]
    fn test_layer_norm_numerical_stability() {
        let ln = LayerNorm::new(3, 1e-8).unwrap();
        let input = vec![1e-10, 2e-10, 3e-10]; // Very small values

        let result = ln.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        // Should not produce NaN or infinite values
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_transformer_layer_creation() {
        let layer = TransformerLayer::new_default(512, 8, 2048);
        assert!(layer.is_ok());

        let tl = layer.unwrap();
        assert_eq!(tl.hidden_size(), 512);
        assert_eq!(tl.num_heads(), 8);
        assert_eq!(tl.intermediate_size(), 2048);
    }

    #[test]
    fn test_transformer_layer_full_creation() {
        let layer = TransformerLayer::new(256, 4, 1024, 1024, 10000.0, 1e-5, 0.1);
        assert!(layer.is_ok());

        let tl = layer.unwrap();
        assert_eq!(tl.hidden_size(), 256);
        assert_eq!(tl.num_heads(), 4);
        assert_eq!(tl.intermediate_size(), 1024);
    }

    #[test]
    fn test_transformer_layer_invalid_config() {
        // Zero hidden size
        assert!(TransformerLayer::new_default(0, 8, 2048).is_err());

        // Zero num heads
        assert!(TransformerLayer::new_default(512, 0, 2048).is_err());

        // Zero intermediate size
        assert!(TransformerLayer::new_default(512, 8, 0).is_err());

        // Hidden size not divisible by num heads
        assert!(TransformerLayer::new_default(513, 8, 2048).is_err());
    }

    #[test]
    fn test_transformer_layer_forward() {
        let layer = TransformerLayer::new_default(64, 4, 256).unwrap();
        let input = vec![vec![1.0; 64], vec![0.5; 64], vec![-0.5; 64]];

        let result = layer.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 3); // Same sequence length
        assert_eq!(output[0].len(), 64); // Same hidden size
        assert_eq!(output[1].len(), 64);
        assert_eq!(output[2].len(), 64);
    }

    #[test]
    fn test_transformer_layer_empty_input() {
        let layer = TransformerLayer::new_default(64, 4, 256).unwrap();
        let input = vec![];

        let result = layer.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_layer_wrong_input_size() {
        let layer = TransformerLayer::new_default(64, 4, 256).unwrap();
        let input = vec![
            vec![1.0; 32], // Wrong size
        ];

        let result = layer.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_layer_single_token() {
        let layer = TransformerLayer::new_default(32, 2, 128).unwrap();
        let input = vec![vec![1.0; 32]];

        let result = layer.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 32);
    }

    #[test]
    fn test_transformer_layer_with_training() {
        let layer = TransformerLayer::new_default(32, 2, 128).unwrap();
        let input = vec![vec![1.0; 32], vec![0.5; 32]];

        // Test training mode
        let result_train = layer.forward_with_training(&input, true);
        assert!(result_train.is_ok());

        // Test inference mode
        let result_infer = layer.forward_with_training(&input, false);
        assert!(result_infer.is_ok());

        let output_train = result_train.unwrap();
        let output_infer = result_infer.unwrap();

        assert_eq!(output_train.len(), 2);
        assert_eq!(output_infer.len(), 2);
        assert_eq!(output_train[0].len(), 32);
        assert_eq!(output_infer[0].len(), 32);
    }

    #[test]
    fn test_residual_connection() {
        let layer = TransformerLayer::new_default(4, 2, 16).unwrap();
        let input = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let residual = vec![vec![0.1, 0.2, 0.3, 0.4]];

        let result = layer.add_residual(&input, &residual);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 4);

        // Check that residual was added correctly
        assert!((output[0][0] - 1.1).abs() < 1e-6);
        assert!((output[0][1] - 2.2).abs() < 1e-6);
        assert!((output[0][2] - 3.3).abs() < 1e-6);
        assert!((output[0][3] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn test_residual_connection_mismatched_lengths() {
        let layer = TransformerLayer::new_default(4, 2, 16).unwrap();
        let input = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let residual = vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7, 0.8]]; // Different sequence length

        let result = layer.add_residual(&input, &residual);
        assert!(result.is_err());
    }

    #[test]
    fn test_residual_connection_mismatched_hidden_size() {
        let layer = TransformerLayer::new_default(4, 2, 16).unwrap();
        let input = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let residual = vec![vec![0.1, 0.2]]; // Different hidden size

        let result = layer.add_residual(&input, &residual);
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_layer_preserves_sequence_length() {
        let layer = TransformerLayer::new_default(32, 4, 128).unwrap();

        // Test different sequence lengths
        for seq_len in [1, 2, 5, 10] {
            let input = vec![vec![1.0; 32]; seq_len];

            let result = layer.forward(&input);
            assert!(result.is_ok());

            let output = result.unwrap();
            assert_eq!(output.len(), seq_len);

            for seq in output {
                assert_eq!(seq.len(), 32);
            }
        }
    }

    #[test]
    fn test_transformer_layer_nonlinearity() {
        let layer = TransformerLayer::new_default(16, 2, 64).unwrap();

        // Test that the transformer layer produces different outputs for different inputs
        // This is a more practical test of functionality rather than strict mathematical nonlinearity
        let input1 = vec![
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ];
        let input2 = vec![
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ];

        let output1 = layer.forward(&input1).unwrap();
        let output2 = layer.forward(&input2).unwrap();

        // Check if outputs are different (they should be due to attention and FFN nonlinearities)
        let mut different = false;
        for (out1, out2) in output1.iter().zip(output2.iter()) {
            for (val1, val2) in out1.iter().zip(out2.iter()) {
                if (val1 - val2).abs() > 1e-6 {
                    different = true;
                    break;
                }
            }
            if different {
                break;
            }
        }

        assert!(
            different,
            "Transformer layer should produce different outputs for different inputs"
        );

        // Also test that the layer doesn't just return the input (should transform it)
        let zero_input = vec![vec![0.0; 16], vec![0.0; 16]];
        let zero_output = layer.forward(&zero_input).unwrap();

        // The output should not be all zeros due to the learned parameters
        let mut has_nonzero = false;
        for seq in &zero_output {
            for &val in seq {
                if val.abs() > 1e-6 {
                    has_nonzero = true;
                    break;
                }
            }
            if has_nonzero {
                break;
            }
        }

        // Note: This might fail if all weights happen to be initialized to produce zero output
        // but that's extremely unlikely with proper random initialization
        assert!(
            has_nonzero,
            "Transformer layer should transform inputs (not just return zeros)"
        );
    }

    #[test]
    fn test_feedforward_nonlinearity() {
        let ff = FeedForward::new(2, 4).unwrap();
        let input1 = vec![1.0, 0.0];
        let input2 = vec![0.0, 1.0];
        let input_sum = vec![1.0, 1.0];

        let output1 = ff.forward(&input1).unwrap();
        let output2 = ff.forward(&input2).unwrap();
        let output_sum = ff.forward(&input_sum).unwrap();

        // Due to nonlinearity, f(a + b) ≠ f(a) + f(b)
        let manual_sum: Vec<f32> = output1
            .iter()
            .zip(output2.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        let mut different = false;
        for (manual, actual) in manual_sum.iter().zip(output_sum.iter()) {
            if (manual - actual).abs() > 1e-6 {
                different = true;
                break;
            }
        }

        assert!(different, "Feed-forward should be nonlinear");
    }
}
