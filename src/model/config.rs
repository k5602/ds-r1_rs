//! # Model Configuration
//!
//! Configuration structures and validation for the DeepSeek R1 model.

use crate::utils::error::{ModelError, Result};
use serde::{Deserialize, Serialize};

/// Attention mechanism choice for transformer layers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AttentionType {
    Standard,
    MLA,
}

/// Feed-forward block choice for transformer layers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FeedForwardType {
    Dense,
    MoE,
}

/// Main configuration for the DeepSeek R1 model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // Basic transformer parameters
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,

    // Architecture toggles
    pub attention_type: AttentionType, // Standard | MLA
    pub ff_type: FeedForwardType,      // Dense | MoE

    // Mixed-depth periodic patterns (optional)
    // When set, periodically replaces the default type:
    // - MLA every `mla_every` layers (e.g., Some(3) -> every 3rd layer uses MLA)
    // - MoE every `moe_every` layers (e.g., Some(4) -> every 4th layer uses MoE)
    pub mla_every: Option<usize>,
    pub moe_every: Option<usize>,

    // MLA specific parameters
    pub kv_compression_ratio: f32,
    pub rope_theta: f32,

    // MoE specific parameters
    pub num_experts: usize,
    pub experts_per_token: usize,

    // Reasoning specific parameters
    pub thinking_token_id: u32,
    pub max_reasoning_steps: usize,

    // Training parameters
    pub dropout_prob: f32,
    pub layer_norm_eps: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 512,
            num_layers: 8,
            num_heads: 8,
            intermediate_size: 2048,
            max_seq_len: 2048,

            // Architecture toggles
            attention_type: AttentionType::Standard,
            ff_type: FeedForwardType::Dense,

            // Mixed-depth periodic patterns (disabled by default)
            mla_every: None,
            moe_every: None,

            // MLA configuration
            kv_compression_ratio: 0.5,
            rope_theta: 10000.0,

            // MoE configuration
            num_experts: 8,
            experts_per_token: 2,

            // Reasoning configuration
            thinking_token_id: 32001, // Special token ID
            max_reasoning_steps: 256,

            // Training configuration
            dropout_prob: 0.1,
            layer_norm_eps: 1e-5,
        }
    }
}

impl ModelConfig {
    /// Create a new configuration with validation
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(ModelError::Config(
                "vocab_size must be greater than 0".to_string(),
            ));
        }

        if self.hidden_size == 0 || self.hidden_size % self.num_heads != 0 {
            return Err(ModelError::Config(
                "hidden_size must be greater than 0 and divisible by num_heads".to_string(),
            ));
        }

        if self.num_layers == 0 {
            return Err(ModelError::Config(
                "num_layers must be greater than 0".to_string(),
            ));
        }

        if self.num_heads == 0 {
            return Err(ModelError::Config(
                "num_heads must be greater than 0".to_string(),
            ));
        }

        // MLA compression ratio must be valid globally;
        // when AttentionType::MLA is selected (or periodically enabled), enforce strict < 1.0
        if self.kv_compression_ratio <= 0.0 || self.kv_compression_ratio > 1.0 {
            return Err(ModelError::Config(
                "kv_compression_ratio must be between 0 and 1".to_string(),
            ));
        }
        if matches!(self.attention_type, AttentionType::MLA) && self.kv_compression_ratio >= 1.0 {
            return Err(ModelError::Config(
                "kv_compression_ratio must be < 1.0 when using MLA".to_string(),
            ));
        }

        // Periodic patterns, if provided, must be >= 1
        if let Some(n) = self.mla_every
            && n == 0
        {
            return Err(ModelError::Config(
                "mla_every must be >= 1 when specified".to_string(),
            ));
        }
        if let Some(n) = self.moe_every
            && n == 0
        {
            return Err(ModelError::Config(
                "moe_every must be >= 1 when specified".to_string(),
            ));
        }

        // MoE configuration sanity
        if self.num_experts == 0 {
            return Err(ModelError::Config(
                "num_experts must be greater than 0".to_string(),
            ));
        }
        if self.experts_per_token == 0 || self.experts_per_token > self.num_experts {
            return Err(ModelError::Config(
                "experts_per_token must be between 1 and num_experts".to_string(),
            ));
        }

        Ok(())
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Get the compressed KV dimension
    pub fn compressed_kv_dim(&self) -> usize {
        (self.hidden_size as f32 * self.kv_compression_ratio) as usize
    }

    /// Save configuration to JSON file
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| ModelError::Config(format!("Failed to serialize config: {}", e)))?;

        std::fs::write(path, json).map_err(ModelError::Io)?;

        Ok(())
    }

    /// Load configuration from JSON file
    pub fn load_from_file(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path).map_err(ModelError::Io)?;

        let config: Self = serde_json::from_str(&json)
            .map_err(|e| ModelError::Config(format!("Failed to deserialize config: {}", e)))?;

        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ModelConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.head_dim(), 64); // 512 / 8
        assert_eq!(config.compressed_kv_dim(), 256); // 512 * 0.5
    }

    #[test]
    fn test_config_validation() {
        let mut config = ModelConfig::default();

        // Test invalid vocab_size
        config.vocab_size = 0;
        assert!(config.validate().is_err());

        // Reset and test invalid hidden_size
        config = ModelConfig::default();
        config.hidden_size = 0;
        assert!(config.validate().is_err());

        // Test hidden_size not divisible by num_heads
        config = ModelConfig::default();
        config.hidden_size = 513; // Not divisible by 8
        assert!(config.validate().is_err());
    }
}
