//! # Transformer Model
//!
//! Main DeepSeek R1 model implementation with transformer layers.

use crate::model::attention::StandardAttentionCache;
use crate::model::config::{AttentionType, FeedForwardType, ModelConfig};
use crate::model::layers::{AttentionKind, FeedForwardKind, LayerNorm, TransformerLayer};
use crate::utils::error::{ModelError, Result};
use rand::Rng;

/// Trainable token embedding owned by the model (exposes mutable weights)
struct TrainableEmbedding {
    weights: Vec<Vec<f32>>,
    vocab_size: usize,
    hidden_size: usize,
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

        // Build transformer layers via factory honoring attention/FF toggles and periodic patterns
        let attention_default = match config.attention_type {
            AttentionType::Standard => AttentionKind::Standard,
            AttentionType::MLA => AttentionKind::MLA,
        };
        let ff_default = match config.ff_type {
            FeedForwardType::Dense => FeedForwardKind::Dense,
            FeedForwardType::MoE => FeedForwardKind::MoE,
        };
        let layers = TransformerLayer::build_layers_mixed(
            config.num_layers,
            config.hidden_size,
            config.num_heads,
            config.intermediate_size,
            config.max_seq_len,
            config.rope_theta,
            config.layer_norm_eps,
            config.dropout_prob,
            attention_default,
            ff_default,
            config.mla_every,
            config.moe_every,
            config.kv_compression_ratio,
            config.num_experts,
            config.experts_per_token,
        )?;

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

    /// Build a mutable parameter registry over all trainable buffers
    pub fn parameters_mut(&mut self) -> crate::model::ParameterRegistryMut<'_> {
        let emb =
            crate::model::collect_rows_mut("embeddings.weight", &mut self.token_embedding.weights);
        let mut head = crate::model::collect_rows_mut("lm_head.weight", &mut self.lm_head.weights);
        let bias = crate::model::single_mut("lm_head.bias", &mut self.lm_head.bias);
        head.push(bias);
        crate::model::registry_from_groups(vec![emb, head])
    }

    /// Parameter metadata (names and lengths) for logging/checkpoint preflight
    pub fn parameters_info(&self) -> Vec<crate::model::ParameterInfo> {
        let mut infos = Vec::new();
        for (i, row) in self.token_embedding.weights.iter().enumerate() {
            infos.push(crate::model::ParameterInfo::new(
                format!("embeddings.weight[{}]", i),
                row.len(),
            ));
        }
        for (i, row) in self.lm_head.weights.iter().enumerate() {
            infos.push(crate::model::ParameterInfo::new(
                format!("lm_head.weight[{}]", i),
                row.len(),
            ));
        }
        infos.push(crate::model::ParameterInfo::new(
            "lm_head.bias",
            self.lm_head.bias.len(),
        ));
        infos
    }

    /// Immutable view of LM head weight matrix: [vocab_size][hidden_size]
    pub fn lm_head_weights(&self) -> &Vec<Vec<f32>> {
        &self.lm_head.weights
    }

    /// Incremental decoding API using per-layer KV caches.
    /// Processes a single input token and returns logits for predicting the next token.
    pub fn forward_next(&mut self, cache: &mut ModelKVCache, token_id: u32) -> Result<Vec<f32>> {
        // note: MLA attention isn't supported for incremental KV decoding in this prototype till now
        if matches!(self.config.attention_type, AttentionType::MLA)
            || self.config.mla_every.is_some()
        {
            return Err(ModelError::Forward(
                "Incremental decoding (KV cache) is not supported when MLA attention is enabled"
                    .to_string(),
            ));
        }
        // Ensure per-layer caches are initialized/sized
        cache.ensure_for_model(self);

        // Determine the current sequence position from the first layer cache
        let position = cache.per_layer.get(0).map(|c| c.seq_len()).unwrap_or(0);

        // 1) Token embedding for the single token
        let emb = self.token_embedding.forward(&[token_id])?;
        let mut h_t = emb
            .get(0)
            .ok_or_else(|| ModelError::Forward("Empty embedding output".to_string()))?
            .clone();

        // 2) Incremental pass through transformer layers with KV cache
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache
                .per_layer
                .get_mut(i)
                .ok_or_else(|| ModelError::Forward("Layer cache missing".to_string()))?;
            h_t = layer.forward_next(layer_cache, &h_t, position, false)?;
        }

        // 3) Final normalization
        let h_final = self.final_norm.forward(&h_t)?;

        // 4) LM head projection to logits for next-token prediction
        let logits = self.lm_head.forward(&h_final)?;

        // 5) Append token to cached token_ids only after successful forward
        cache.token_ids.push(token_id);

        Ok(logits)
    }

    // ===== Telemetry helpers for MLA/MoE integration =====

    /// Per-layer attention kinds as strings: "Standard" or "MLA"
    pub fn layer_attention_kinds(&self) -> Vec<&'static str> {
        self.layers
            .iter()
            .map(|l| match l.attention_kind() {
                crate::model::layers::AttentionKind::Standard => "Standard",
                crate::model::layers::AttentionKind::MLA => "MLA",
            })
            .collect()
    }

    /// Per-layer feed-forward kinds as strings: "Dense" or "MoE"
    pub fn layer_ff_kinds(&self) -> Vec<&'static str> {
        self.layers
            .iter()
            .map(|l| match l.ff_kind() {
                crate::model::layers::FeedForwardKind::Dense => "Dense",
                crate::model::layers::FeedForwardKind::MoE => "MoE",
            })
            .collect()
    }

    /// Per-layer MLA compression stats, if MLA is active on that layer.
    /// Each entry is Some((avg_ratio, memory_savings_percent)) or None if not MLA.
    pub fn layer_mla_compression_stats(&self) -> Vec<Option<(f32, f32)>> {
        self.layers
            .iter()
            .map(|l| l.mla_compression_stats())
            .collect()
    }

    /// Per-layer MoE expert utilization distributions, if MoE is active on that layer.
    /// Each entry is Some(utilization_vec) or None if not MoE.
    pub fn layer_moe_utilization(&self) -> Vec<Option<Vec<f32>>> {
        self.layers
            .iter()
            .map(|l| l.moe_expert_utilization())
            .collect()
    }

    /// Per-layer MoE load balance loss (variance), if MoE is active on that layer.
    /// Each entry is Some(variance) or None if not MoE.
    pub fn layer_moe_load_balance_loss(&self) -> Vec<Option<f32>> {
        self.layers
            .iter()
            .map(|l| l.moe_load_balance_loss())
            .collect()
    }

    /// Reset MoE load balancer statistics on all layers that use MoE.
    /// Returns the count of layers that were reset.
    pub fn reset_all_moe_load_balancers(&mut self) -> usize {
        let mut count = 0usize;
        for layer in &self.layers {
            if layer.moe_reset_load_balancer() {
                count += 1;
            }
        }
        count
    }
}

/// Model-level KV cache holding the token prefix and per-layer attention caches.
/// The per-layer caches are sized according to the model's layer count and each
/// cache is initialized with the corresponding number of heads.
pub struct ModelKVCache {
    pub token_ids: Vec<u32>,
    pub per_layer: Vec<StandardAttentionCache>,
}

impl ModelKVCache {
    /// Create a new empty cache aligned with the provided model
    pub fn new(model: &DeepSeekR1Model) -> Self {
        let mut per_layer = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            per_layer.push(StandardAttentionCache::new(layer.num_heads()));
        }
        Self {
            token_ids: Vec::new(),
            per_layer,
        }
    }

    /// Ensure the internal per-layer caches match the model's layers
    pub fn ensure_for_model(&mut self, model: &DeepSeekR1Model) {
        if self.per_layer.len() != model.layers.len() {
            self.per_layer.clear();
            for layer in &model.layers {
                self.per_layer
                    .push(StandardAttentionCache::new(layer.num_heads()));
            }
        }
    }

    /// Reset the cached prefix and clear all per-layer caches
    pub fn clear(&mut self) {
        self.token_ids.clear();
        for cache in &mut self.per_layer {
            cache.clear();
        }
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

    #[test]
    fn test_incremental_cache_creation() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config.clone()).unwrap();

        // Just verify we can create a cache and it has the right structure
        let cache = super::ModelKVCache::new(&model);
        assert_eq!(cache.per_layer.len(), config.num_layers);
        assert!(cache.token_ids.is_empty());

        // Verify cache can be cleared
        let mut cache = cache;
        cache.clear();
        assert!(cache.token_ids.is_empty());
        assert_eq!(cache.per_layer.len(), config.num_layers);
    }

    #[test]
    fn test_kv_cache_growth_and_position() {
        let config = ModelConfig::default();
        let mut model = DeepSeekR1Model::new(config.clone()).unwrap();

        let seq = [10u32, 20u32, 30u32, 40u32, 50u32];
        let mut cache = super::ModelKVCache::new(&model);

        // Initially empty
        assert_eq!(cache.token_ids.len(), 0);
        assert!(
            !cache.per_layer.is_empty(),
            "per-layer caches not initialized"
        );
        let mut expected_len = 0usize;

        for (idx, &tok) in seq.iter().enumerate() {
            // Before feeding next token, all layer caches should report expected_len
            for layer_cache in &cache.per_layer {
                assert_eq!(
                    layer_cache.seq_len(),
                    expected_len,
                    "seq_len should equal number of tokens already fed"
                );
            }

            // Feed token
            let _ = model
                .forward_next(&mut cache, tok)
                .expect("forward_next failed");
            expected_len += 1;

            // After feeding, seq_len should increment by 1 on all layers
            for (lidx, layer_cache) in cache.per_layer.iter().enumerate() {
                assert_eq!(
                    layer_cache.seq_len(),
                    expected_len,
                    "layer {} seq_len mismatch at step {}",
                    lidx,
                    idx
                );
            }

            // Token ids in cache should match the fed prefix
            assert_eq!(cache.token_ids.len(), expected_len);
            assert_eq!(&cache.token_ids[..], &seq[..=idx]);
        }

        // Clear and verify reset
        cache.clear();
        assert_eq!(cache.token_ids.len(), 0);
        for layer_cache in &cache.per_layer {
            assert_eq!(layer_cache.seq_len(), 0);
        }
    }
}
