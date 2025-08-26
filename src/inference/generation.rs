//! # Text Generation
//!
//! Text generation utilities and configuration.

use serde::{Deserialize, Serialize};

/// Generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub stop_tokens: Vec<u32>,
    pub repetition_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            stop_tokens: vec![],
            repetition_penalty: 1.0,
        }
    }
}

/// Output from text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationOutput {
    pub text: String,
    pub tokens_generated: usize,
    pub stop_reason: StopReason,
    pub generation_time_ms: u64,
}

/// Reason why generation stopped
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StopReason {
    MaxTokens,
    StopToken,
    EndOfSequence,
    Error(String),
}

impl GenerationOutput {
    /// Create a new generation output
    pub fn new(text: String, tokens_generated: usize, stop_reason: StopReason) -> Self {
        Self {
            text,
            tokens_generated,
            stop_reason,
            generation_time_ms: 0,
        }
    }
    
    /// Set generation time
    pub fn with_time(mut self, time_ms: u64) -> Self {
        self.generation_time_ms = time_ms;
        self
    }
    
    /// Check if generation was successful
    pub fn is_success(&self) -> bool {
        !matches!(self.stop_reason, StopReason::Error(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 256);
        assert_eq!(config.temperature, 1.0);
        assert!(config.stop_tokens.is_empty());
    }

    #[test]
    fn test_generation_output() {
        let output = GenerationOutput::new(
            "Hello world".to_string(),
            2,
            StopReason::MaxTokens,
        );
        
        assert_eq!(output.text, "Hello world");
        assert_eq!(output.tokens_generated, 2);
        assert!(output.is_success());
    }

    #[test]
    fn test_generation_output_with_time() {
        let output = GenerationOutput::new(
            "Test".to_string(),
            1,
            StopReason::EndOfSequence,
        ).with_time(100);
        
        assert_eq!(output.generation_time_ms, 100);
    }

    #[test]
    fn test_generation_output_error() {
        let output = GenerationOutput::new(
            "".to_string(),
            0,
            StopReason::Error("Test error".to_string()),
        );
        
        assert!(!output.is_success());
    }
}