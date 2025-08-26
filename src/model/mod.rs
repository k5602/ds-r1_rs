//! # Model Module
//!
//! Core model components including transformer architecture, attention mechanisms,
//! and mixture-of-experts implementation.

pub mod config;
pub mod transformer;
pub mod attention;
pub mod moe;
pub mod embeddings;
pub mod layers;

// Re-export key types
pub use config::ModelConfig;
pub use transformer::DeepSeekR1Model;
pub use attention::{MLAAttention, StandardAttention};
pub use moe::{MoELayer, Expert};
pub use embeddings::{TokenEmbedding, PositionalEmbedding};
pub use layers::{TransformerLayer, FeedForward, LayerNorm};