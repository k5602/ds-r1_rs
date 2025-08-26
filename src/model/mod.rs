//! # Model Module
//!
//! Core model components including transformer architecture, attention mechanisms,
//! and mixture-of-experts implementation.

pub mod attention;
pub mod config;
pub mod embeddings;
pub mod layers;
pub mod moe;
pub mod transformer;

// Re-export key types
pub use attention::{Linear, MLAAttention, StandardAttention};
pub use config::ModelConfig;
pub use embeddings::{RotaryEmbedding, TokenEmbedding};
pub use layers::{FeedForward, LayerNorm, TransformerLayer};
pub use moe::{Expert, MoELayer};
pub use transformer::DeepSeekR1Model;
