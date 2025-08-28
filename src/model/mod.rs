//! # Model Module
//!
//! Core model components including transformer architecture, attention mechanisms,
//! and mixture-of-experts implementation.

pub mod attention;
pub mod config;
pub mod embeddings;
pub mod layers;
pub mod moe;
pub mod parameters;
pub mod transformer;

// Re-export key types
pub use attention::{Linear, MLAAttention, StandardAttention};
pub use config::ModelConfig;
pub use embeddings::{RotaryEmbedding, TokenEmbedding};
pub use layers::{FeedForward, LayerNorm, TransformerLayer};
pub use moe::{Expert, MoELayer};
pub use parameters::{
    Parameter, ParameterInfo, ParameterMut, ParameterRef, ParameterRegistryMut, collect_rows_mut,
    collect_rows_ref, registry_from_groups, single_mut, single_ref,
};
pub use transformer::DeepSeekR1Model;
