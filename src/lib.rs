//! # DeepSeek R1 Rust Implementation
//!
//! A prototype implementation of DeepSeek R1-inspired reasoning model in Rust.
//! This library provides core components for transformer architecture, multi-head
//! latent attention (MLA), mixture-of-experts (MoE), and reasoning capabilities.

pub mod model;
pub mod training;
pub mod inference;
pub mod utils;

// Re-export commonly used types and functions
pub use model::{
    config::ModelConfig,
    transformer::DeepSeekR1Model,
};

pub use inference::{
    engine::InferenceEngine,
    reasoning::ReasoningOutput,
};

pub use training::{
    trainer::BasicTrainer,
    data::TrainingExample,
};

pub use utils::{
    error::{ModelError, Result},
    math::MathUtils,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default model configuration for quick prototyping
pub fn default_config() -> ModelConfig {
    ModelConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_default_config() {
        let config = default_config();
        assert!(config.vocab_size > 0);
        assert!(config.hidden_size > 0);
        assert!(config.num_layers > 0);
    }
}