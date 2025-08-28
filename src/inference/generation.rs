//! # Text Generation
//!
//! Text generation utilities and configuration.

use crate::inference::reasoning::{ReasoningOutput, TokenReasoningProcessor};
use crate::inference::sampling::{Sampler, SamplingConfig};
use crate::model::DeepSeekR1Model;
use crate::utils::error::{ModelError, Result};
use crate::utils::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::time::Instant;

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

/// Text generator for autoregressive generation
pub struct TextGenerator {
    sampler: Sampler,
}

impl TextGenerator {
    /// Create a new text generator
    pub fn new(sampling_config: SamplingConfig) -> Self {
        Self {
            sampler: Sampler::new(sampling_config),
        }
    }

    /// Generate text using autoregressive sampling
    pub fn generate(
        &mut self,
        model: &mut DeepSeekR1Model,
        tokenizer: &Tokenizer,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<GenerationOutput> {
        let start_time = Instant::now();

        // Encode the prompt
        let mut token_ids = tokenizer.encode(prompt)?;
        let original_length = token_ids.len();
        let mut tokens_generated = 0;

        // Generation loop
        for _ in 0..config.max_tokens {
            // Forward pass through model
            let logits = model.forward(&token_ids)?;
            
            // Get logits for the last position (next token prediction)
            let vocab_size = tokenizer.vocab_size();
            if logits.len() < vocab_size {
                return Err(ModelError::Forward(
                    "Model output size doesn't match vocabulary size".to_string(),
                ));
            }
            
            let next_token_logits = &logits[logits.len() - vocab_size..];

            // Sample next token based on temperature
            let next_token = if config.temperature <= 0.0 {
                self.sampler.sample_greedy(next_token_logits)?
            } else {
                self.sampler.sample_temperature(next_token_logits)?
            };

            // Check for stop conditions
            if config.stop_tokens.contains(&next_token) {
                let generated_text = self.decode_generated_tokens(
                    tokenizer,
                    &token_ids[original_length..],
                )?;
                let elapsed = start_time.elapsed().as_millis() as u64;
                return Ok(GenerationOutput::new(
                    generated_text,
                    tokens_generated,
                    StopReason::StopToken,
                ).with_time(elapsed));
            }

            // Check for EOS token
            if let Ok(eos_id) = tokenizer.eos_token_id() {
                if next_token == eos_id {
                    let generated_text = self.decode_generated_tokens(
                        tokenizer,
                        &token_ids[original_length..],
                    )?;
                    let elapsed = start_time.elapsed().as_millis() as u64;
                    return Ok(GenerationOutput::new(
                        generated_text,
                        tokens_generated,
                        StopReason::EndOfSequence,
                    ).with_time(elapsed));
                }
            }

            // Add token to sequence
            token_ids.push(next_token);
            tokens_generated += 1;
        }

        // Reached max tokens
        let generated_text = self.decode_generated_tokens(
            tokenizer,
            &token_ids[original_length..],
        )?;
        let elapsed = start_time.elapsed().as_millis() as u64;
        Ok(GenerationOutput::new(
            generated_text,
            tokens_generated,
            StopReason::MaxTokens,
        ).with_time(elapsed))
    }

    /// Generate text with caching for efficiency
    pub fn generate_with_cache(
        &mut self,
        model: &mut DeepSeekR1Model,
        tokenizer: &Tokenizer,
        prompt: &str,
        config: &GenerationConfig,
        _cache: &mut GenerationCache,
    ) -> Result<GenerationOutput> {
        // For now, just call the regular generate method
        // TODO: Implement proper KV caching in future iterations
        self.generate(model, tokenizer, prompt, config)
    }

    /// Generate text with reasoning awareness
    pub fn generate_with_reasoning(
        &mut self,
        model: &mut DeepSeekR1Model,
        tokenizer: &Tokenizer,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<ReasoningOutput> {
        let _start_time = Instant::now();

        // Get thinking token IDs
        let think_start_id = tokenizer.think_start_token_id()?;
        let think_end_id = tokenizer.think_end_token_id()?;

        // Create reasoning processor
        let mut reasoning_processor = TokenReasoningProcessor::new(think_start_id, think_end_id);

        // Encode the prompt
        let mut token_ids = tokenizer.encode(prompt)?;
        let _original_length = token_ids.len();
        let _tokens_generated = 0;

        // Generation loop with reasoning awareness
        for _ in 0..config.max_tokens {
            // Forward pass through model
            let logits = model.forward(&token_ids)?;
            
            // Get logits for the last position (next token prediction)
            let vocab_size = tokenizer.vocab_size();
            if logits.len() < vocab_size {
                return Err(ModelError::Forward(
                    "Model output size doesn't match vocabulary size".to_string(),
                ));
            }
            
            let next_token_logits = &logits[logits.len() - vocab_size..];

            // Sample next token
            let next_token = if config.temperature <= 0.0 {
                self.sampler.sample_greedy(next_token_logits)?
            } else {
                self.sampler.sample_temperature(next_token_logits)?
            };

            // Decode token for reasoning processor
            let token_text = tokenizer.decode(&[next_token])?;
            
            // Process token with reasoning engine
            reasoning_processor.process_generation_token(next_token, &token_text)?;

            // Check for stop conditions
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            // Check for EOS token
            if let Ok(eos_id) = tokenizer.eos_token_id() {
                if next_token == eos_id {
                    break;
                }
            }

            // Add token to sequence
            token_ids.push(next_token);
            // tokens_generated += 1; // Commented out since variable is unused
        }

        // Get final reasoning output
        Ok(reasoning_processor.get_reasoning_output())
    }

    /// Generate text with reasoning mode detection
    pub fn generate_with_reasoning_detection(
        &mut self,
        model: &mut DeepSeekR1Model,
        tokenizer: &Tokenizer,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<(GenerationOutput, Option<ReasoningOutput>)> {
        // First generate normally
        let generation_output = self.generate(model, tokenizer, prompt, config)?;
        
        // Check if the generated text contains thinking tokens
        let think_start_id = tokenizer.think_start_token_id()?;
        let think_end_id = tokenizer.think_end_token_id()?;
        
        if generation_output.text.contains("<think>") || generation_output.text.contains("</think>") {
            // Parse reasoning from the generated text
            let mut reasoning_processor = TokenReasoningProcessor::new(think_start_id, think_end_id);
            
            // Process the generated text token by token (simplified)
            let tokens = tokenizer.encode(&generation_output.text)?;
            for token_id in tokens {
                let token_text = tokenizer.decode(&[token_id])?;
                reasoning_processor.process_generation_token(token_id, &token_text)?;
            }
            
            let reasoning_output = reasoning_processor.get_reasoning_output();
            Ok((generation_output, Some(reasoning_output)))
        } else {
            Ok((generation_output, None))
        }
    }

    /// Generate text with structured reasoning output
    pub fn generate_structured_reasoning(
        &mut self,
        model: &mut DeepSeekR1Model,
        tokenizer: &Tokenizer,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<ReasoningOutput> {
        // Add reasoning prompt if not present
        let reasoning_prompt = if prompt.contains("<think>") {
            prompt.to_string()
        } else {
            format!("{} <think>Let me think about this step by step.</think>", prompt)
        };

        self.generate_with_reasoning(model, tokenizer, &reasoning_prompt, config)
    }

    /// Decode only the newly generated tokens
    fn decode_generated_tokens(&self, tokenizer: &Tokenizer, token_ids: &[u32]) -> Result<String> {
        if token_ids.is_empty() {
            return Ok(String::new());
        }
        tokenizer.decode(token_ids)
    }
}

/// Cache for generation state (placeholder for future KV caching)
pub struct GenerationCache {
    // TODO: Implement KV cache in future iterations
    _placeholder: (),
}

impl GenerationCache {
    /// Create a new generation cache
    pub fn new() -> Self {
        Self { _placeholder: () }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        // TODO: Implement cache clearing
    }
}

impl Default for GenerationCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::ModelConfig;
    use crate::utils::tokenizer::TokenizerConfig;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 256);
        assert_eq!(config.temperature, 1.0);
        assert!(config.stop_tokens.is_empty());
    }

    #[test]
    fn test_generation_output() {
        let output = GenerationOutput::new("Hello world".to_string(), 2, StopReason::MaxTokens);

        assert_eq!(output.text, "Hello world");
        assert_eq!(output.tokens_generated, 2);
        assert!(output.is_success());
    }

    #[test]
    fn test_generation_output_with_time() {
        let output =
            GenerationOutput::new("Test".to_string(), 1, StopReason::EndOfSequence).with_time(100);

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

    #[test]
    fn test_text_generator_creation() {
        let sampling_config = SamplingConfig::default();
        let _generator = TextGenerator::new(sampling_config);
    }

    #[test]
    fn test_generation_cache() {
        let mut cache = GenerationCache::new();
        cache.clear(); // Should not panic
        
        let default_cache = GenerationCache::default();
        assert_eq!(std::mem::size_of_val(&default_cache), std::mem::size_of::<()>());
    }

    #[test]
    fn test_decode_generated_tokens() {
        let sampling_config = SamplingConfig::default();
        let generator = TextGenerator::new(sampling_config);
        let tokenizer_config = TokenizerConfig::default();
        let tokenizer = Tokenizer::new(tokenizer_config).unwrap();

        // Test empty tokens
        let result = generator.decode_generated_tokens(&tokenizer, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "");
    }

    #[test]
    fn test_reasoning_generation_methods_exist() {
        let sampling_config = SamplingConfig::default();
        let mut generator = TextGenerator::new(sampling_config);
        let model_config = ModelConfig::default();
        let mut model = DeepSeekR1Model::new(model_config).unwrap();
        let tokenizer_config = TokenizerConfig::default();
        let tokenizer = Tokenizer::new(tokenizer_config).unwrap();
        let gen_config = GenerationConfig::default();

        // These should not panic (though they may return errors due to unimplemented model)
        let _result1 = generator.generate_with_reasoning(&mut model, &tokenizer, "test", &gen_config);
        let _result2 = generator.generate_with_reasoning_detection(&mut model, &tokenizer, "test", &gen_config);
        let _result3 = generator.generate_structured_reasoning(&mut model, &tokenizer, "test", &gen_config);
    }

    // Note: Full generation testing requires a working model implementation
    // which will be available in later tasks
}
