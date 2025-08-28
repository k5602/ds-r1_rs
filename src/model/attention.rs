//! # Attention Mechanisms
//!
//! Implementation of Multi-head Latent Attention (MLA) and standard attention.

use crate::model::embeddings::RotaryEmbedding;
use crate::utils::error::{ModelError, Result};
use rand::Rng;

/// LoRA-style low-rank decomposition for KV compression
pub struct LoRACompression {
    /// Down projection: [hidden_size] -> [compressed_dim]
    down_proj: Linear,
    /// Up projection: [compressed_dim] -> [hidden_size]
    up_proj: Linear,
    /// Layer normalization for compressed representation
    norm: crate::model::layers::LayerNorm,
    hidden_size: usize,
    compressed_dim: usize,
}

impl LoRACompression {
    /// Create a new LoRA compression layer
    pub fn new(hidden_size: usize, compressed_dim: usize, layer_norm_eps: f32) -> Result<Self> {
        if hidden_size == 0 || compressed_dim == 0 {
            return Err(ModelError::Config(
                "hidden_size and compressed_dim must be greater than 0".to_string(),
            ));
        }

        if compressed_dim >= hidden_size {
            return Err(ModelError::Config(
                "compressed_dim must be less than hidden_size for compression".to_string(),
            ));
        }

        let down_proj = Linear::new(hidden_size, compressed_dim)?;
        let up_proj = Linear::new(compressed_dim, hidden_size)?;
        let norm = crate::model::layers::LayerNorm::new(compressed_dim, layer_norm_eps)?;

        Ok(Self {
            down_proj,
            up_proj,
            norm,
            hidden_size,
            compressed_dim,
        })
    }

    /// Compress input
    pub fn compress(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.hidden_size {
            return Err(ModelError::Forward(format!(
                "Input size {} doesn't match hidden_size {}",
                input.len(),
                self.hidden_size
            )));
        }

        let compressed = self.down_proj.forward(input)?;
        let normalized = self.norm.forward(&compressed)?;
        Ok(normalized)
    }

    /// Decompress input: [compressed_dim] -> [hidden_size]
    pub fn decompress(&self, compressed: &[f32]) -> Result<Vec<f32>> {
        if compressed.len() != self.compressed_dim {
            return Err(ModelError::Forward(format!(
                "Compressed size {} doesn't match compressed_dim {}",
                compressed.len(),
                self.compressed_dim
            )));
        }

        self.up_proj.forward(compressed)
    }

    /// Compress batch: [batch_size, hidden_size] -> [batch_size, compressed_dim]
    pub fn compress_batch(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(input.len());
        for input_vec in input {
            let compressed = self.compress(input_vec)?;
            output.push(compressed);
        }
        Ok(output)
    }

    /// Decompress batch: [batch_size, compressed_dim] -> [batch_size, hidden_size]
    pub fn decompress_batch(&self, compressed: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(compressed.len());
        for compressed_vec in compressed {
            let decompressed = self.decompress(compressed_vec)?;
            output.push(decompressed);
        }
        Ok(output)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.compressed_dim as f32 / self.hidden_size as f32
    }

    /// memory savings
    pub fn memory_savings(&self) -> f32 {
        (1.0 - self.compression_ratio()) * 100.0
    }
}

/// Multi-head Latent Attention implementation with compressed KV representations
pub struct MLAAttention {
    /// Query projection (full size)
    q_proj: Linear,
    /// Key compression using LoRA-style decomposition
    k_compression: LoRACompression,
    /// Value compression using LoRA-style decomposition
    v_compression: LoRACompression,
    /// Output projection
    o_proj: Linear,
    /// Rotary position embeddings
    rope: RotaryEmbedding,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Hidden dimension
    hidden_size: usize,
    /// Compressed KV dimension
    compressed_kv_dim: usize,
    /// Dropout probability
    dropout_prob: f32,
}

impl MLAAttention {
    /// Create a new MLA attention layer
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        compression_ratio: f32,
        max_seq_len: usize,
        rope_theta: f32,
        layer_norm_eps: f32,
    ) -> Result<Self> {
        if hidden_size == 0 || num_heads == 0 {
            return Err(ModelError::Config(
                "hidden_size and num_heads must be greater than 0".to_string(),
            ));
        }

        if hidden_size % num_heads != 0 {
            return Err(ModelError::Config(
                "hidden_size must be divisible by num_heads".to_string(),
            ));
        }

        if compression_ratio <= 0.0 || compression_ratio >= 1.0 {
            return Err(ModelError::Config(
                "compression_ratio must be between 0 and 1".to_string(),
            ));
        }

        let head_dim = hidden_size / num_heads;
        let compressed_kv_dim = (hidden_size as f32 * compression_ratio) as usize;
        let compressed_kv_dim = compressed_kv_dim.max(1);
        let q_proj = Linear::new(hidden_size, hidden_size)?;
        let k_compression = LoRACompression::new(hidden_size, compressed_kv_dim, layer_norm_eps)?;
        let v_compression = LoRACompression::new(hidden_size, compressed_kv_dim, layer_norm_eps)?;
        let o_proj = Linear::new(hidden_size, hidden_size)?;
        let rotary_dim_per_head = head_dim / 2;
        let rope = RotaryEmbedding::new(rotary_dim_per_head, max_seq_len, rope_theta)?;

        Ok(Self {
            q_proj,
            k_compression,
            v_compression,
            o_proj,
            rope,
            num_heads,
            head_dim,
            hidden_size,
            compressed_kv_dim,
            dropout_prob: 0.1,
        })
    }

    ///MLA attention with default parameters
    pub fn new_default(
        hidden_size: usize,
        num_heads: usize,
        compression_ratio: f32,
    ) -> Result<Self> {
        Self::new(
            hidden_size,
            num_heads,
            compression_ratio,
            2048,    // max_seq_len
            10000.0, // rope_theta
            1e-5,    // layer_norm_eps
        )
    }

    /// Get compression statistics
    pub fn compression_stats(&self) -> (f32, f32) {
        let k_ratio = self.k_compression.compression_ratio();
        let v_ratio = self.v_compression.compression_ratio();
        let avg_ratio = (k_ratio + v_ratio) / 2.0;
        let memory_savings = (1.0 - avg_ratio) * 100.0;
        (avg_ratio, memory_savings)
    }

    /// Split tensor into rotary and non-rotary components for MLA
    fn split_rotary_components(
        &self,
        tensor: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let seq_len = tensor.len();
        if seq_len == 0 {
            return Err(ModelError::Forward(
                "Input tensor cannot be empty".to_string(),
            ));
        }

        // For MLA, we typically apply rotary embeddings to half the dimensions
        let rotary_dim = self.head_dim / 2;
        let _non_rotary_dim = self.head_dim - rotary_dim;

        let mut rotary_parts = Vec::with_capacity(seq_len);
        let mut non_rotary_parts = Vec::with_capacity(seq_len);

        for seq_vec in tensor {
            if seq_vec.len() != self.hidden_size {
                return Err(ModelError::Forward(format!(
                    "Tensor dimension {} doesn't match hidden_size {}",
                    seq_vec.len(),
                    self.hidden_size
                )));
            }

            // Split each head's dimensions
            let mut rotary_seq = Vec::with_capacity(self.hidden_size / 2);
            let mut non_rotary_seq = Vec::with_capacity(self.hidden_size / 2);

            for head in 0..self.num_heads {
                let head_start = head * self.head_dim;

                // Rotary part: first half of each head's dimensions
                for i in 0..rotary_dim {
                    rotary_seq.push(seq_vec[head_start + i]);
                }

                // Non-rotary part: second half of each head's dimensions
                for i in rotary_dim..self.head_dim {
                    non_rotary_seq.push(seq_vec[head_start + i]);
                }
            }

            rotary_parts.push(rotary_seq);
            non_rotary_parts.push(non_rotary_seq);
        }

        Ok((rotary_parts, non_rotary_parts))
    }

    /// Recombine rotary and non-rotary components after rotary transformation
    fn recombine_rotary_components(
        &self,
        rotary_parts: &[Vec<f32>],
        non_rotary_parts: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        if rotary_parts.len() != non_rotary_parts.len() {
            return Err(ModelError::Forward(
                "Rotary and non-rotary parts must have same sequence length".to_string(),
            ));
        }

        let seq_len = rotary_parts.len();
        let rotary_dim = self.head_dim / 2;
        let non_rotary_dim = self.head_dim - rotary_dim;

        let mut result = Vec::with_capacity(seq_len);

        for (rotary_seq, non_rotary_seq) in rotary_parts.iter().zip(non_rotary_parts.iter()) {
            if rotary_seq.len() != self.num_heads * rotary_dim {
                return Err(ModelError::Forward(
                    "Rotary part dimension mismatch".to_string(),
                ));
            }
            if non_rotary_seq.len() != self.num_heads * non_rotary_dim {
                return Err(ModelError::Forward(
                    "Non-rotary part dimension mismatch".to_string(),
                ));
            }

            let mut combined_seq = vec![0.0; self.hidden_size];

            for head in 0..self.num_heads {
                let head_start = head * self.head_dim;
                let rotary_head_start = head * rotary_dim;
                let non_rotary_head_start = head * non_rotary_dim;

                // Copy rotary part
                for i in 0..rotary_dim {
                    combined_seq[head_start + i] = rotary_seq[rotary_head_start + i];
                }

                // Copy non-rotary part
                for i in 0..non_rotary_dim {
                    combined_seq[head_start + rotary_dim + i] =
                        non_rotary_seq[non_rotary_head_start + i];
                }
            }

            result.push(combined_seq);
        }

        Ok(result)
    }

    /// Apply rotary embeddings to MLA query and key tensors
    /// This splits tensors, applies rotary to appropriate parts, and recombines
    pub fn apply_mla_rotary(
        &self,
        queries: &[Vec<f32>],
        keys: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        if queries.len() != keys.len() {
            return Err(ModelError::Forward(
                "Queries and keys must have same sequence length".to_string(),
            ));
        }

        // Split queries into rotary and non-rotary components
        let (q_rotary, q_non_rotary) = self.split_rotary_components(queries)?;
        let (k_rotary, k_non_rotary) = self.split_rotary_components(keys)?;

        // Apply rotary embeddings to rotary components
        // We need to reshape for per-head application
        let mut q_rotary_transformed = Vec::with_capacity(q_rotary.len());
        let mut k_rotary_transformed = Vec::with_capacity(k_rotary.len());

        let rotary_dim_per_head = self.head_dim / 2;

        for (pos, (q_seq, k_seq)) in q_rotary.iter().zip(k_rotary.iter()).enumerate() {
            let mut q_transformed_seq = Vec::with_capacity(q_seq.len());
            let mut k_transformed_seq = Vec::with_capacity(k_seq.len());

            // Apply rotary to each head separately
            for head in 0..self.num_heads {
                let head_start = head * rotary_dim_per_head;
                let head_end = head_start + rotary_dim_per_head;

                let q_head_slice = &q_seq[head_start..head_end];
                let k_head_slice = &k_seq[head_start..head_end];

                let q_rotated = self.rope.apply_rotary(q_head_slice, pos)?;
                let k_rotated = self.rope.apply_rotary(k_head_slice, pos)?;

                q_transformed_seq.extend(q_rotated);
                k_transformed_seq.extend(k_rotated);
            }

            q_rotary_transformed.push(q_transformed_seq);
            k_rotary_transformed.push(k_transformed_seq);
        }

        // Recombine rotary and non-rotary components
        let q_final = self.recombine_rotary_components(&q_rotary_transformed, &q_non_rotary)?;
        let k_final = self.recombine_rotary_components(&k_rotary_transformed, &k_non_rotary)?;

        Ok((q_final, k_final))
    }

    /// Apply causal mask to attention scores
    fn apply_causal_mask(&self, scores: &mut Vec<Vec<f32>>) {
        let seq_len = scores.len();
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[i][j] = f32::NEG_INFINITY;
            }
        }
    }

    /// Apply softmax to attention scores
    fn softmax(&self, scores: &mut Vec<Vec<f32>>) -> Result<()> {
        for row in scores {
            // Find max for numerical stability
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Subtract max and exponentiate
            let mut sum = 0.0;
            for val in row.iter_mut() {
                *val = (*val - max_val).exp();
                sum += *val;
            }

            // Normalize
            if sum > 0.0 {
                for val in row {
                    *val /= sum;
                }
            }
        }

        Ok(())
    }

    /// Reshape tensor for multi-head attention: [seq_len, hidden_size] -> [num_heads, seq_len, head_dim]
    fn reshape_for_attention(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<Vec<f32>>>> {
        let seq_len = input.len();
        let mut output = vec![vec![vec![0.0; self.head_dim]; seq_len]; self.num_heads];

        for (seq_idx, seq_vec) in input.iter().enumerate() {
            if seq_vec.len() != self.hidden_size {
                return Err(ModelError::Forward("Input dimension mismatch".to_string()));
            }

            for head in 0..self.num_heads {
                let start_idx = head * self.head_dim;
                let end_idx = start_idx + self.head_dim;
                output[head][seq_idx].copy_from_slice(&seq_vec[start_idx..end_idx]);
            }
        }

        Ok(output)
    }

    /// Reshape back from multi-head: [num_heads, seq_len, head_dim] -> [seq_len, hidden_size]
    fn reshape_from_attention(&self, input: &[Vec<Vec<f32>>]) -> Result<Vec<Vec<f32>>> {
        let seq_len = input[0].len();
        let mut output = vec![vec![0.0; self.hidden_size]; seq_len];

        for seq_idx in 0..seq_len {
            for head in 0..self.num_heads {
                let start_idx = head * self.head_dim;
                let end_idx = start_idx + self.head_dim;
                output[seq_idx][start_idx..end_idx].copy_from_slice(&input[head][seq_idx]);
            }
        }

        Ok(output)
    }

    /// Calculate memory usage for KV cache (in number of float32 elements)
    pub fn calculate_memory_usage(&self, seq_len: usize) -> (usize, usize) {
        // Standard attention: full KV representations
        let standard_kv_memory = seq_len * self.hidden_size * 2; // K and V

        // MLA: compressed KV representations
        let mla_kv_memory = seq_len * self.compressed_kv_dim * 2; // Compressed K and V

        (standard_kv_memory, mla_kv_memory)
    }

    /// Get memory reduction percentage compared to standard attention
    pub fn memory_reduction_percentage(&self, seq_len: usize) -> f32 {
        let (standard_memory, mla_memory) = self.calculate_memory_usage(seq_len);
        ((standard_memory - mla_memory) as f32 / standard_memory as f32) * 100.0
    }

    /// Forward pass through MLA attention with compressed KV representations
    /// Input: [seq_len, hidden_size], Output: [seq_len, hidden_size]
    pub fn forward(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let seq_len = input.len();
        if seq_len == 0 {
            return Err(ModelError::Forward(
                "Input sequence cannot be empty".to_string(),
            ));
        }

        // Step 1: Project to Q, compress K and V
        let queries = self.q_proj.forward_batch(input)?;

        // Compress keys and values using LoRA-style compression
        let compressed_keys = self.k_compression.compress_batch(input)?;
        let compressed_values = self.v_compression.compress_batch(input)?;

        // Step 2: Decompress K and V for attention computation
        let keys = self.k_compression.decompress_batch(&compressed_keys)?;
        let values = self.v_compression.decompress_batch(&compressed_values)?;

        // Step 3: Apply rotary embeddings to queries and keys
        let (q_rotated, k_rotated) = self.apply_mla_rotary(&queries, &keys)?;

        // Step 4: Reshape for multi-head attention
        let q_heads = self.reshape_for_attention(&q_rotated)?;
        let k_heads = self.reshape_for_attention(&k_rotated)?;
        let v_heads = self.reshape_for_attention(&values)?;

        // Step 5: Compute attention for each head
        let mut output_heads = vec![vec![vec![0.0; self.head_dim]; seq_len]; self.num_heads];

        for head in 0..self.num_heads {
            // Compute attention scores: Q @ K^T
            let mut scores = vec![vec![0.0; seq_len]; seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut score = 0.0;
                    for k in 0..self.head_dim {
                        score += q_heads[head][i][k] * k_heads[head][j][k];
                    }
                    scores[i][j] = score / (self.head_dim as f32).sqrt();
                }
            }

            // Apply causal mask
            self.apply_causal_mask(&mut scores);

            // Apply softmax
            self.softmax(&mut scores)?;

            // Apply attention to values: scores @ V
            for i in 0..seq_len {
                for k in 0..self.head_dim {
                    let mut sum = 0.0;
                    for j in 0..seq_len {
                        sum += scores[i][j] * v_heads[head][j][k];
                    }
                    output_heads[head][i][k] = sum;
                }
            }
        }

        // Step 6: Reshape back and apply output projection
        let concatenated = self.reshape_from_attention(&output_heads)?;
        let output = self.o_proj.forward_batch(&concatenated)?;

        Ok(output)
    }

    /// Get number of heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get compressed KV dimension
    pub fn compressed_kv_dim(&self) -> usize {
        self.compressed_kv_dim
    }
}

/// Simple linear layer for projections
pub struct Linear {
    /// Weight matrix: [out_features, in_features]
    weights: Vec<Vec<f32>>,
    /// Bias vector: [out_features]
    bias: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// Create a new linear layer with random initialization
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(ModelError::Config(
                "Features must be greater than 0".to_string(),
            ));
        }

        let mut rng = rand::rng();
        let std_dev = (1.0 / in_features as f32).sqrt();

        // Initialize weights with Xavier/Glorot initialization
        let mut weights = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            let mut row = Vec::with_capacity(in_features);
            for _ in 0..in_features {
                row.push(rng.random::<f32>() * 2.0 * std_dev - std_dev);
            }
            weights.push(row);
        }

        // Initialize bias to small random values instead of zero as zeros failed because
        // `Linear` weights are properly randomized and nonzero, but when the input is all zeros,
        // the output from the `Linear` layer is also all zeros.
        // This causes the attention scores to be uniform, leading to uniform attention weights after softmax.
        // By initializing the bias to small random values, we ensure that even with zero input,
        // the output from the `Linear` layer is not uniform, allowing the model to learn.
        // This made me understand how initialization impacts model performance - zero bias creates
        // degenerate attention matrices with eigenvalues at λ₁ = n⁻¹, maximizing entropy H(A) = log(n)
        // and providing zero information bits about relationships. Random bias breaks this symmetry,
        // preventing pathological gradients with high correlation across heads and enabling each
        // head to specialize through divergent flows. Small perturbations create a bifurcation
        // point that fundamentally alters training dynamics and avoids rank-deficient subspaces.
        // Thanks for this test problem from the past.
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

    /// Forward pass: output = input @ weight.T + bias
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.in_features {
            return Err(ModelError::Forward(format!(
                "Input size {} doesn't match in_features {}",
                input.len(),
                self.in_features
            )));
        }

        let mut output = vec![0.0; self.out_features];

        for (i, output_val) in output.iter_mut().enumerate() {
            let mut sum = self.bias[i];
            for (j, &input_val) in input.iter().enumerate() {
                sum += self.weights[i][j] * input_val;
            }
            *output_val = sum;
        }
        Ok(output)
    }

    /// Forward pass for batch: [batch_size, in_features] -> [batch_size, out_features]
    pub fn forward_batch(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(input.len());

        for input_vec in input {
            let output_vec = self.forward(input_vec)?;
            output.push(output_vec);
        }

        Ok(output)
    }
}

/// Standard multi-head attention implementation
pub struct StandardAttention {
    /// Query projection
    q_proj: Linear,
    /// Key projection
    k_proj: Linear,
    /// Value projection
    v_proj: Linear,
    /// Output projection
    o_proj: Linear,
    /// Rotary position embeddings
    rope: RotaryEmbedding,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Hidden dimension
    hidden_size: usize,
    /// Dropout probability
    dropout_prob: f32,
}

impl StandardAttention {
    /// Create a new standard attention layer
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        rope_theta: f32,
    ) -> Result<Self> {
        if hidden_size == 0 || num_heads == 0 {
            return Err(ModelError::Config(
                "hidden_size and num_heads must be greater than 0".to_string(),
            ));
        }

        if hidden_size % num_heads != 0 {
            return Err(ModelError::Config(
                "hidden_size must be divisible by num_heads".to_string(),
            ));
        }

        let head_dim = hidden_size / num_heads;

        // Create projection layers
        let q_proj = Linear::new(hidden_size, hidden_size)?;
        let k_proj = Linear::new(hidden_size, hidden_size)?;
        let v_proj = Linear::new(hidden_size, hidden_size)?;
        let o_proj = Linear::new(hidden_size, hidden_size)?;

        // Create rotary embeddings
        let rope = RotaryEmbedding::new(head_dim, max_seq_len, rope_theta)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            num_heads,
            head_dim,
            hidden_size,
            dropout_prob: 0.1,
        })
    }

    /// Apply causal mask to attention scores
    fn apply_causal_mask(&self, scores: &mut Vec<Vec<f32>>) {
        let seq_len = scores.len();
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[i][j] = f32::NEG_INFINITY;
            }
        }
    }

    /// Apply softmax to attention scores
    fn softmax(&self, scores: &mut Vec<Vec<f32>>) -> Result<()> {
        for row in scores {
            // Find max for numerical stability
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Subtract max and exponentiate
            let mut sum = 0.0;
            for val in row.iter_mut() {
                *val = (*val - max_val).exp();
                sum += *val;
            }

            // Normalize
            if sum > 0.0 {
                for val in row {
                    *val /= sum;
                }
            }
        }

        Ok(())
    }

    /// Reshape tensor for multi-head attention: [seq_len, hidden_size] -> [num_heads, seq_len, head_dim]
    fn reshape_for_attention(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<Vec<f32>>>> {
        let seq_len = input.len();
        let mut output = vec![vec![vec![0.0; self.head_dim]; seq_len]; self.num_heads];

        for (seq_idx, seq_vec) in input.iter().enumerate() {
            if seq_vec.len() != self.hidden_size {
                return Err(ModelError::Forward("Input dimension mismatch".to_string()));
            }

            for head in 0..self.num_heads {
                let start_idx = head * self.head_dim;
                let end_idx = start_idx + self.head_dim;
                output[head][seq_idx].copy_from_slice(&seq_vec[start_idx..end_idx]);
            }
        }

        Ok(output)
    }
    fn reshape_from_attention(&self, input: &[Vec<Vec<f32>>]) -> Result<Vec<Vec<f32>>> {
        let seq_len = input[0].len();
        let mut output = vec![vec![0.0; self.hidden_size]; seq_len];

        for seq_idx in 0..seq_len {
            for head in 0..self.num_heads {
                let start_idx = head * self.head_dim;
                let end_idx = start_idx + self.head_dim;
                output[seq_idx][start_idx..end_idx].copy_from_slice(&input[head][seq_idx]);
            }
        }

        Ok(output)
    }

    /// Forward pass through standard attention
    /// Input: [seq_len, hidden_size], Output: [seq_len, hidden_size]
    pub fn forward(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let seq_len = input.len();
        if seq_len == 0 {
            return Err(ModelError::Forward(
                "Input sequence cannot be empty".to_string(),
            ));
        }

        // Project to Q, K, V
        let queries = self.q_proj.forward_batch(input)?;
        let keys = self.k_proj.forward_batch(input)?;
        let values = self.v_proj.forward_batch(input)?;

        // Reshape for multi-head attention
        let q_heads = self.reshape_for_attention(&queries)?;
        let k_heads = self.reshape_for_attention(&keys)?;
        let v_heads = self.reshape_for_attention(&values)?;

        // Apply attention for each head
        let mut output_heads = vec![vec![vec![0.0; self.head_dim]; seq_len]; self.num_heads];

        for head in 0..self.num_heads {
            // Apply rotary embeddings to queries and keys
            let q_rotated = self.rope.apply_rotary_batch(&q_heads[head])?;
            let k_rotated = self.rope.apply_rotary_batch(&k_heads[head])?;

            // Compute attention scores: Q @ K^T
            let mut scores = vec![vec![0.0; seq_len]; seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut score = 0.0;
                    for k in 0..self.head_dim {
                        score += q_rotated[i][k] * k_rotated[j][k];
                    }
                    scores[i][j] = score / (self.head_dim as f32).sqrt();
                }
            }

            // Apply causal mask
            self.apply_causal_mask(&mut scores);

            // Apply softmax
            self.softmax(&mut scores)?;

            // Apply attention to values: scores @ V
            for i in 0..seq_len {
                for k in 0..self.head_dim {
                    let mut sum = 0.0;
                    for j in 0..seq_len {
                        sum += scores[i][j] * v_heads[head][j][k];
                    }
                    output_heads[head][i][k] = sum;
                }
            }
        }

        // Reshape back and apply output projection
        let concatenated = self.reshape_from_attention(&output_heads)?;
        let output = self.o_proj.forward_batch(&concatenated)?;

        Ok(output)
    }

    /// Get number of heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer_creation() {
        let linear = Linear::new(512, 256);
        assert!(linear.is_ok());

        let layer = linear.unwrap();
        assert_eq!(layer.in_features, 512);
        assert_eq!(layer.out_features, 256);
    }

    #[test]
    fn test_linear_layer_forward() {
        let linear = Linear::new(4, 2).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let result = linear.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_linear_layer_batch() {
        let linear = Linear::new(3, 2).unwrap();
        let input = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let result = linear.forward_batch(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 2);
        assert_eq!(output[1].len(), 2);
    }

    #[test]
    fn test_linear_layer_invalid_input() {
        let linear = Linear::new(4, 2).unwrap();
        let input = vec![1.0, 2.0]; // Wrong size

        let result = linear.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_standard_attention_creation() {
        let attention = StandardAttention::new(512, 8, 2048, 10000.0);
        assert!(attention.is_ok());

        let attn = attention.unwrap();
        assert_eq!(attn.num_heads(), 8);
        assert_eq!(attn.head_dim(), 64);
    }

    #[test]
    fn test_standard_attention_invalid_config() {
        let attention = StandardAttention::new(513, 8, 2048, 10000.0);
        assert!(attention.is_err());

        // Zero values
        let attention = StandardAttention::new(0, 8, 2048, 10000.0);
        assert!(attention.is_err());

        let attention = StandardAttention::new(512, 0, 2048, 10000.0);
        assert!(attention.is_err());
    }

    #[test]
    fn test_standard_attention_forward() {
        let attention = StandardAttention::new(64, 4, 128, 10000.0).unwrap();
        let input = vec![
            vec![1.0; 64], // First token
            vec![0.5; 64], // Second token
            vec![0.0; 64], // Third token
        ];

        let result = attention.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 3); // Same sequence length
        assert_eq!(output[0].len(), 64); // Same hidden size
        assert_eq!(output[1].len(), 64);
        assert_eq!(output[2].len(), 64);
    }

    #[test]
    fn test_standard_attention_empty_input() {
        let attention = StandardAttention::new(64, 4, 128, 10000.0).unwrap();
        let input = vec![];

        let result = attention.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_reshape_operations() {
        let attention = StandardAttention::new(64, 4, 128, 10000.0).unwrap();
        let input = vec![vec![1.0; 64], vec![2.0; 64]];

        // Test reshape for attention
        let reshaped = attention.reshape_for_attention(&input);
        assert!(reshaped.is_ok());

        let heads = reshaped.unwrap();
        assert_eq!(heads.len(), 4); // num_heads
        assert_eq!(heads[0].len(), 2); // seq_len
        assert_eq!(heads[0][0].len(), 16); // head_dim

        // Test reshape back
        let back = attention.reshape_from_attention(&heads);
        assert!(back.is_ok());

        let reconstructed = back.unwrap();
        assert_eq!(reconstructed.len(), 2); // seq_len
        assert_eq!(reconstructed[0].len(), 64); // hidden_size
    }

    #[test]
    fn test_softmax_function() {
        let attention = StandardAttention::new(64, 4, 128, 10000.0).unwrap();
        let mut scores = vec![vec![1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0]];

        let result = attention.softmax(&mut scores);
        assert!(result.is_ok());

        // Check that each row sums to approximately 1.0
        for row in &scores {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Check that probabilities are positive
        for row in &scores {
            for &val in row {
                assert!(val >= 0.0);
            }
        }
    }

    #[test]
    fn test_causal_mask() {
        let attention = StandardAttention::new(64, 4, 128, 10000.0).unwrap();
        let mut scores = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        attention.apply_causal_mask(&mut scores);

        // Check upper triangular part is masked
        assert_eq!(scores[0][1], f32::NEG_INFINITY);
        assert_eq!(scores[0][2], f32::NEG_INFINITY);
        assert_eq!(scores[1][2], f32::NEG_INFINITY);

        // Check lower triangular and diagonal are unchanged
        assert_eq!(scores[0][0], 1.0);
        assert_eq!(scores[1][0], 4.0);
        assert_eq!(scores[1][1], 5.0);
        assert_eq!(scores[2][0], 7.0);
        assert_eq!(scores[2][1], 8.0);
        assert_eq!(scores[2][2], 9.0);
    }

    #[test]
    fn test_lora_compression_creation() {
        let compression = LoRACompression::new(512, 256, 1e-5);
        assert!(compression.is_ok());

        let comp = compression.unwrap();
        assert_eq!(comp.compression_ratio(), 0.5);
        assert_eq!(comp.memory_savings(), 50.0);
    }

    #[test]
    fn test_lora_compression_invalid_config() {
        // Zero dimensions
        assert!(LoRACompression::new(0, 256, 1e-5).is_err());
        assert!(LoRACompression::new(512, 0, 1e-5).is_err());

        // Compressed dim >= hidden size (no compression)
        assert!(LoRACompression::new(512, 512, 1e-5).is_err());
        assert!(LoRACompression::new(512, 600, 1e-5).is_err());
    }

    #[test]
    fn test_lora_compression_forward() {
        let compression = LoRACompression::new(64, 32, 1e-5).unwrap();
        let input = vec![1.0; 64];

        // Test compression
        let compressed = compression.compress(&input);
        assert!(compressed.is_ok());
        let comp_vec = compressed.unwrap();
        assert_eq!(comp_vec.len(), 32);

        // Test decompression
        let decompressed = compression.decompress(&comp_vec);
        assert!(decompressed.is_ok());
        let decomp_vec = decompressed.unwrap();
        assert_eq!(decomp_vec.len(), 64);
    }

    #[test]
    fn test_lora_compression_batch() {
        let compression = LoRACompression::new(32, 16, 1e-5).unwrap();
        let input = vec![vec![1.0; 32], vec![0.5; 32], vec![-0.5; 32]];

        // Test batch compression
        let compressed = compression.compress_batch(&input);
        assert!(compressed.is_ok());
        let comp_batch = compressed.unwrap();
        assert_eq!(comp_batch.len(), 3);
        assert_eq!(comp_batch[0].len(), 16);

        // Test batch decompression
        let decompressed = compression.decompress_batch(&comp_batch);
        assert!(decompressed.is_ok());
        let decomp_batch = decompressed.unwrap();
        assert_eq!(decomp_batch.len(), 3);
        assert_eq!(decomp_batch[0].len(), 32);
    }

    #[test]
    fn test_lora_compression_wrong_input_size() {
        let compression = LoRACompression::new(64, 32, 1e-5).unwrap();

        // Wrong input size for compression
        let wrong_input = vec![1.0; 32];
        assert!(compression.compress(&wrong_input).is_err());

        // Wrong input size for decompression
        let wrong_compressed = vec![1.0; 16];
        assert!(compression.decompress(&wrong_compressed).is_err());
    }

    #[test]
    fn test_lora_compression_ratios() {
        // Test different compression ratios
        let comp_25 = LoRACompression::new(100, 25, 1e-5).unwrap();
        assert_eq!(comp_25.compression_ratio(), 0.25);
        assert_eq!(comp_25.memory_savings(), 75.0);

        let comp_75 = LoRACompression::new(100, 75, 1e-5).unwrap();
        assert_eq!(comp_75.compression_ratio(), 0.75);
        assert_eq!(comp_75.memory_savings(), 25.0);
    }

    #[test]
    fn test_lora_compression_information_preservation() {
        let compression = LoRACompression::new(16, 8, 1e-5).unwrap();

        // Test that different inputs produce different compressed representations
        let input1 = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let input2 = vec![
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let comp1 = compression.compress(&input1).unwrap();
        let comp2 = compression.compress(&input2).unwrap();

        // Compressed representations should be different
        let mut different = false;
        for (c1, c2) in comp1.iter().zip(comp2.iter()) {
            if (c1 - c2).abs() > 1e-6 {
                different = true;
                break;
            }
        }
        assert!(
            different,
            "Different inputs should produce different compressed representations"
        );
    }

    #[test]
    fn test_mla_creation() {
        let mla = MLAAttention::new_default(512, 8, 0.5);
        assert!(mla.is_ok());

        let attention = mla.unwrap();
        assert_eq!(attention.num_heads(), 8);
        assert_eq!(attention.head_dim(), 64);
        assert_eq!(attention.compressed_kv_dim(), 256);

        let (ratio, savings) = attention.compression_stats();
        assert_eq!(ratio, 0.5);
        assert_eq!(savings, 50.0);
    }

    #[test]
    fn test_mla_invalid_config() {
        // Invalid compression ratio
        assert!(MLAAttention::new_default(512, 8, 0.0).is_err());
        assert!(MLAAttention::new_default(512, 8, 1.0).is_err());
        assert!(MLAAttention::new_default(512, 8, 1.5).is_err());

        // Invalid dimensions
        assert!(MLAAttention::new_default(0, 8, 0.5).is_err());
        assert!(MLAAttention::new_default(512, 0, 0.5).is_err());

        // Hidden size not divisible by num_heads
        assert!(MLAAttention::new_default(513, 8, 0.5).is_err());
    }

    #[test]
    fn test_mla_compression_efficiency() {
        // Test different compression ratios
        let mla_25 = MLAAttention::new_default(512, 8, 0.25).unwrap();
        let mla_50 = MLAAttention::new_default(512, 8, 0.5).unwrap();
        let mla_75 = MLAAttention::new_default(512, 8, 0.75).unwrap();

        let (ratio_25, savings_25) = mla_25.compression_stats();
        let (ratio_50, savings_50) = mla_50.compression_stats();
        let (ratio_75, savings_75) = mla_75.compression_stats();

        assert_eq!(ratio_25, 0.25);
        assert_eq!(savings_25, 75.0);
        assert_eq!(ratio_50, 0.5);
        assert_eq!(savings_50, 50.0);
        assert_eq!(ratio_75, 0.75);
        assert_eq!(savings_75, 25.0);

        // Higher compression should result in smaller compressed dimensions
        assert!(mla_25.compressed_kv_dim() < mla_50.compressed_kv_dim());
        assert!(mla_50.compressed_kv_dim() < mla_75.compressed_kv_dim());
    }

    #[test]
    fn test_mla_memory_comparison() {
        let hidden_size = 512;
        let num_heads = 8;

        // Standard attention uses full KV representations
        let standard = StandardAttention::new(hidden_size, num_heads, 2048, 10000.0).unwrap();

        // MLA uses compressed KV representations
        let mla = MLAAttention::new_default(hidden_size, num_heads, 0.5).unwrap();

        // Calculate memory usage for KV cache (simplified)
        let seq_len = 100;
        let standard_kv_memory = seq_len * hidden_size * 2; // K and V
        let mla_kv_memory = seq_len * mla.compressed_kv_dim() * 2; // Compressed K and V

        let memory_reduction =
            (standard_kv_memory - mla_kv_memory) as f32 / standard_kv_memory as f32;

        // Should achieve approximately 50% memory reduction
        assert!(memory_reduction > 0.4 && memory_reduction < 0.6);
    }

    #[test]
    fn test_mla_split_rotary_components() {
        let mla = MLAAttention::new_default(64, 4, 0.5).unwrap(); // head_dim = 16
        let input = vec![
            vec![1.0; 64], // First sequence position
            vec![0.5; 64], // Second sequence position
        ];

        let result = mla.split_rotary_components(&input);
        assert!(result.is_ok());

        let (rotary_parts, non_rotary_parts) = result.unwrap();

        // Each head contributes head_dim/2 = 8 dimensions to rotary part
        // With 4 heads: 4 * 8 = 32 dimensions total for rotary
        assert_eq!(rotary_parts.len(), 2); // seq_len
        assert_eq!(rotary_parts[0].len(), 32); // 4 heads * 8 rotary dims per head
        assert_eq!(non_rotary_parts.len(), 2); // seq_len
        assert_eq!(non_rotary_parts[0].len(), 32); // 4 heads * 8 non-rotary dims per head
    }

    #[test]
    fn test_mla_recombine_rotary_components() {
        let mla = MLAAttention::new_default(64, 4, 0.5).unwrap();
        let input = vec![vec![1.0; 64], vec![0.5; 64]];

        // Split and then recombine
        let (rotary_parts, non_rotary_parts) = mla.split_rotary_components(&input).unwrap();
        let result = mla.recombine_rotary_components(&rotary_parts, &non_rotary_parts);

        assert!(result.is_ok());
        let recombined = result.unwrap();

        assert_eq!(recombined.len(), 2); // seq_len
        assert_eq!(recombined[0].len(), 64); // hidden_size
        assert_eq!(recombined[1].len(), 64); // hidden_size

        // Check that recombination preserves original values
        for (orig, recom) in input.iter().zip(recombined.iter()) {
            for (o, r) in orig.iter().zip(recom.iter()) {
                assert!((o - r).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_mla_split_recombine_different_inputs() {
        let mla = MLAAttention::new_default(32, 2, 0.5).unwrap(); // head_dim = 16

        // Create input with distinct values to verify correct splitting/recombining
        let mut input = vec![vec![0.0; 32]];
        for i in 0..32 {
            input[0][i] = i as f32;
        }

        let (rotary_parts, non_rotary_parts) = mla.split_rotary_components(&input).unwrap();
        let recombined = mla
            .recombine_rotary_components(&rotary_parts, &non_rotary_parts)
            .unwrap();

        // Verify exact reconstruction
        for (orig, recom) in input[0].iter().zip(recombined[0].iter()) {
            assert_eq!(*orig, *recom);
        }
    }

    #[test]
    fn test_mla_apply_rotary_embeddings() {
        let mla = MLAAttention::new_default(64, 4, 0.5).unwrap();
        let queries = vec![vec![1.0; 64], vec![0.5; 64]];
        let keys = vec![vec![0.8; 64], vec![0.3; 64]];

        let result = mla.apply_mla_rotary(&queries, &keys);
        assert!(result.is_ok());

        let (q_rotated, k_rotated) = result.unwrap();

        assert_eq!(q_rotated.len(), 2); // seq_len
        assert_eq!(q_rotated[0].len(), 64); // hidden_size
        assert_eq!(k_rotated.len(), 2); // seq_len
        assert_eq!(k_rotated[0].len(), 64); // hidden_size

        // Rotated tensors should be different from originals (due to rotary transformation)
        let mut q_different = false;
        let mut k_different = false;

        // Check transformation at position 1 (not position 0, which is identity)
        for (orig, rot) in queries[1].iter().zip(q_rotated[1].iter()) {
            if (orig - rot).abs() > 1e-6 {
                q_different = true;
                break;
            }
        }

        for (orig, rot) in keys[1].iter().zip(k_rotated[1].iter()) {
            if (orig - rot).abs() > 1e-6 {
                k_different = true;
                break;
            }
        }

        assert!(
            q_different,
            "Queries should be transformed by rotary embeddings"
        );
        assert!(
            k_different,
            "Keys should be transformed by rotary embeddings"
        );
    }

    #[test]
    fn test_mla_rotary_position_dependence() {
        let mla = MLAAttention::new_default(32, 2, 0.5).unwrap();
        let queries = vec![
            vec![1.0; 32], // Position 0
            vec![1.0; 32], // Position 1 - same input, different position
        ];
        let keys = vec![
            vec![0.5; 32], // Position 0
            vec![0.5; 32], // Position 1 - same input, different position
        ];

        let (q_rotated, k_rotated) = mla.apply_mla_rotary(&queries, &keys).unwrap();

        // Different positions should produce different outputs even with same input
        let mut q_pos_different = false;
        let mut k_pos_different = false;

        for (pos0, pos1) in q_rotated[0].iter().zip(q_rotated[1].iter()) {
            if (pos0 - pos1).abs() > 1e-6 {
                q_pos_different = true;
                break;
            }
        }

        for (pos0, pos1) in k_rotated[0].iter().zip(k_rotated[1].iter()) {
            if (pos0 - pos1).abs() > 1e-6 {
                k_pos_different = true;
                break;
            }
        }

        assert!(
            q_pos_different,
            "Different positions should produce different query outputs"
        );
        assert!(
            k_pos_different,
            "Different positions should produce different key outputs"
        );
    }

    #[test]
    fn test_mla_rotary_mismatched_sequence_lengths() {
        let mla = MLAAttention::new_default(32, 2, 0.5).unwrap();
        let queries = vec![vec![1.0; 32]]; // Length 1
        let keys = vec![vec![0.5; 32], vec![0.3; 32]]; // Length 2

        let result = mla.apply_mla_rotary(&queries, &keys);
        assert!(result.is_err());
    }

    #[test]
    fn test_mla_split_empty_input() {
        let mla = MLAAttention::new_default(32, 2, 0.5).unwrap();
        let empty_input = vec![];

        let result = mla.split_rotary_components(&empty_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_mla_split_wrong_dimension() {
        let mla = MLAAttention::new_default(32, 2, 0.5).unwrap();
        let wrong_input = vec![vec![1.0; 16]]; // Wrong hidden size

        let result = mla.split_rotary_components(&wrong_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_mla_recombine_mismatched_lengths() {
        let mla = MLAAttention::new_default(32, 2, 0.5).unwrap();
        let rotary_parts = vec![vec![1.0; 16]]; // Length 1
        let non_rotary_parts = vec![vec![0.5; 16], vec![0.3; 16]]; // Length 2

        let result = mla.recombine_rotary_components(&rotary_parts, &non_rotary_parts);
        assert!(result.is_err());
    }

    #[test]
    fn test_mla_rotary_preserves_non_rotary_parts() {
        let mla = MLAAttention::new_default(64, 4, 0.5).unwrap();

        // Create input where we can track which parts should remain unchanged
        let mut queries = vec![vec![0.0; 64]];
        let mut keys = vec![vec![0.0; 64]];

        // Set non-rotary parts to specific values
        let head_dim = 16;
        let rotary_dim = head_dim / 2;

        for head in 0..4 {
            let head_start = head * head_dim;
            // Set non-rotary part (second half of each head) to distinctive values
            for i in rotary_dim..head_dim {
                queries[0][head_start + i] = (head * 10 + i) as f32;
                keys[0][head_start + i] = (head * 10 + i + 100) as f32;
            }
        }

        let original_queries = queries.clone();
        let original_keys = keys.clone();

        let (q_rotated, k_rotated) = mla.apply_mla_rotary(&queries, &keys).unwrap();

        // Check that non-rotary parts are preserved
        for head in 0..4 {
            let head_start = head * head_dim;
            for i in rotary_dim..head_dim {
                let orig_q = original_queries[0][head_start + i];
                let rot_q = q_rotated[0][head_start + i];
                let orig_k = original_keys[0][head_start + i];
                let rot_k = k_rotated[0][head_start + i];

                assert_eq!(orig_q, rot_q, "Non-rotary query parts should be preserved");
                assert_eq!(orig_k, rot_k, "Non-rotary key parts should be preserved");
            }
        }
    }

    #[test]
    fn test_mla_forward_complete() {
        let mla = MLAAttention::new_default(64, 4, 0.5).unwrap();
        let input = vec![vec![1.0; 64], vec![0.5; 64], vec![-0.5; 64]];

        let result = mla.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 3); // Same sequence length
        assert_eq!(output[0].len(), 64); // Same hidden size
        assert_eq!(output[1].len(), 64);
        assert_eq!(output[2].len(), 64);

        // Output should be different from input (due to attention transformation)
        let mut different = false;
        for (inp, out) in input.iter().zip(output.iter()) {
            for (i, o) in inp.iter().zip(out.iter()) {
                if (i - o).abs() > 1e-6 {
                    different = true;
                    break;
                }
            }
            if different {
                break;
            }
        }
        assert!(different, "MLA should transform the input");
    }

    #[test]
    fn test_mla_forward_single_token() {
        let mla = MLAAttention::new_default(32, 2, 0.5).unwrap();
        let input = vec![vec![1.0; 32]];

        let result = mla.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 32);
    }

    #[test]
    fn test_mla_forward_empty_input() {
        let mla = MLAAttention::new_default(32, 2, 0.5).unwrap();
        let input = vec![];

        let result = mla.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_mla_forward_wrong_input_size() {
        let mla = MLAAttention::new_default(64, 4, 0.5).unwrap();
        let input = vec![vec![1.0; 32]]; // Wrong hidden size

        let result = mla.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_mla_memory_usage_calculation() {
        let mla = MLAAttention::new_default(512, 8, 0.5).unwrap();
        let seq_len = 100;

        let (standard_memory, mla_memory) = mla.calculate_memory_usage(seq_len);

        // Standard: seq_len * hidden_size * 2 (K and V)
        assert_eq!(standard_memory, 100 * 512 * 2);

        // MLA: seq_len * compressed_kv_dim * 2 (compressed K and V)
        assert_eq!(mla_memory, 100 * 256 * 2); // 256 = 512 * 0.5

        // MLA should use less memory
        assert!(mla_memory < standard_memory);
    }

    #[test]
    fn test_mla_memory_reduction_percentage() {
        let mla = MLAAttention::new_default(512, 8, 0.5).unwrap();
        let seq_len = 100;

        let reduction = mla.memory_reduction_percentage(seq_len);

        // With 50% compression, should achieve 50% memory reduction
        assert!((reduction - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_mla_memory_reduction_different_ratios() {
        let mla_25 = MLAAttention::new_default(512, 8, 0.25).unwrap();
        let mla_50 = MLAAttention::new_default(512, 8, 0.5).unwrap();
        let mla_75 = MLAAttention::new_default(512, 8, 0.75).unwrap();

        let seq_len = 100;
        let reduction_25 = mla_25.memory_reduction_percentage(seq_len);
        let reduction_50 = mla_50.memory_reduction_percentage(seq_len);
        let reduction_75 = mla_75.memory_reduction_percentage(seq_len);

        // Higher compression should result in higher memory reduction
        assert!(reduction_25 > reduction_50);
        assert!(reduction_50 > reduction_75);

        // Check approximate values
        assert!((reduction_25 - 75.0).abs() < 1.0);
        assert!((reduction_50 - 50.0).abs() < 1.0);
        assert!((reduction_75 - 25.0).abs() < 1.0);
    }

    #[test]
    fn test_mla_vs_standard_attention_memory_comparison() {
        let hidden_size = 512;
        let num_heads = 8;
        let seq_len = 200;

        let standard = StandardAttention::new(hidden_size, num_heads, 2048, 10000.0).unwrap();
        let mla = MLAAttention::new_default(hidden_size, num_heads, 0.5).unwrap();

        // Calculate memory usage for both
        let standard_kv_memory = seq_len * hidden_size * 2; // Full KV cache
        let (calculated_standard, mla_memory) = mla.calculate_memory_usage(seq_len);

        assert_eq!(standard_kv_memory, calculated_standard);
        assert!(mla_memory < standard_kv_memory);

        let memory_savings =
            ((standard_kv_memory - mla_memory) as f32 / standard_kv_memory as f32) * 100.0;
        assert!(memory_savings > 40.0 && memory_savings < 60.0); // Should be around 50%
    }

    #[test]
    fn test_mla_forward_causal_attention() {
        let mla = MLAAttention::new_default(32, 2, 0.5).unwrap();
        let input = vec![
            vec![1.0; 32], // Position 0
            vec![2.0; 32], // Position 1
            vec![3.0; 32], // Position 2
        ];

        let output = mla.forward(&input).unwrap();

        // Due to causal masking, each position should only attend to previous positions
        // This is hard to test directly, but we can verify the output is reasonable
        assert_eq!(output.len(), 3);
        for seq in &output {
            assert_eq!(seq.len(), 32);
            // Check that output values are finite
            for &val in seq {
                assert!(val.is_finite());
            }
        }
    }

    #[test]
    fn test_mla_forward_different_sequence_lengths() {
        let mla = MLAAttention::new_default(64, 4, 0.5).unwrap();

        // Test different sequence lengths
        for seq_len in [1, 2, 5, 10] {
            let input = vec![vec![1.0; 64]; seq_len];

            let result = mla.forward(&input);
            assert!(result.is_ok(), "Failed for sequence length {}", seq_len);

            let output = result.unwrap();
            assert_eq!(output.len(), seq_len);

            for seq in output {
                assert_eq!(seq.len(), 64);
                // Verify output is not all zeros or NaN
                let sum: f32 = seq.iter().sum();
                assert!(sum.is_finite() && sum.abs() > 1e-6);
            }
        }
    }

    #[test]
    fn test_mla_compression_vs_decompression_cycle() {
        let mla = MLAAttention::new_default(64, 4, 0.5).unwrap();
        let input = vec![vec![1.0; 64], vec![0.5; 64]];

        // Test that compression and decompression work in the forward pass
        let result = mla.forward(&input);
        assert!(result.is_ok());

        // The forward pass internally compresses and decompresses K,V
        // If this works without error, the compression cycle is functional
        let output = result.unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 64);
    }

    #[test]
    fn test_attention_single_token() {
        let attention = StandardAttention::new(32, 2, 64, 10000.0).unwrap();
        let input = vec![vec![1.0; 32]]; // Single token

        let result = attention.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 32);
    }
}
