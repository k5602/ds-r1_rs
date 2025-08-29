//! # Inference Engine
//!
//! Main inference engine for text generation and reasoning.

use crate::inference::generation::{
    GenerationCache, GenerationConfig, GenerationOutput, TextGenerator,
};
use crate::inference::reasoning::ReasoningOutput;
use crate::inference::sampling::SamplingConfig;
use crate::model::DeepSeekR1Model;
use crate::utils::error::Result;
use crate::utils::tokenizer::{Tokenizer, TokenizerConfig};
use serde::{Deserialize, Serialize};

/// Types of problems that can be solved
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProblemType {
    Mathematical,
    Logical,
    CodeAnalysis,
    General,
}

/// Mathematical solution output with structured steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathSolutionOutput {
    pub problem: String,
    pub reasoning_steps: Vec<String>,
    pub final_answer: Option<String>,
    pub confidence: f32,
}

/// Code explanation output with structured analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExplanationOutput {
    pub original_code: String,
    pub language: Option<String>,
    pub reasoning_steps: Vec<String>,
    pub summary: String,
    pub confidence: f32,
}

/// Logical reasoning solution output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalSolutionOutput {
    pub problem: String,
    pub reasoning_steps: Vec<String>,
    pub conclusion: String,
    pub confidence: f32,
}

/// Main inference engine
pub struct InferenceEngine {
    model: DeepSeekR1Model,
    tokenizer: Tokenizer,
    text_generator: TextGenerator,
    generation_cache: GenerationCache,
    default_config: GenerationConfig,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(model: DeepSeekR1Model) -> Result<Self> {
        let tokenizer_config = TokenizerConfig {
            vocab_size: model.config().vocab_size,
            ..TokenizerConfig::default()
        };
        let tokenizer = Tokenizer::new(tokenizer_config)?;

        let sampling_config = SamplingConfig::default();
        let text_generator = TextGenerator::new(sampling_config);

        let generation_cache = GenerationCache::new();
        let default_config = GenerationConfig::default();

        Ok(Self {
            model,
            tokenizer,
            text_generator,
            generation_cache,
            default_config,
        })
    }

    /// Create a new inference engine with custom configurations
    pub fn with_configs(
        model: DeepSeekR1Model,
        tokenizer_config: TokenizerConfig,
        sampling_config: SamplingConfig,
        generation_config: GenerationConfig,
    ) -> Result<Self> {
        // Ensure tokenizer vocab size matches model vocab size
        let tokenizer_config = TokenizerConfig {
            vocab_size: model.config().vocab_size,
            ..tokenizer_config
        };
        let tokenizer = Tokenizer::new(tokenizer_config)?;
        let text_generator = TextGenerator::new(sampling_config);
        let generation_cache = GenerationCache::new();

        Ok(Self {
            model,
            tokenizer,
            text_generator,
            generation_cache,
            default_config: generation_config,
        })
    }

    /// Generate text from a prompt using default configuration
    pub fn generate_text(&mut self, prompt: &str) -> Result<String> {
        let output = self.generate_text_with_config(prompt, &self.default_config.clone())?;
        Ok(output.text)
    }

    /// Generate text from a prompt with custom configuration
    pub fn generate_text_with_config(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<GenerationOutput> {
        self.text_generator.generate_with_cache(
            &mut self.model,
            &self.tokenizer,
            prompt,
            config,
            &mut self.generation_cache,
        )
    }

    /// Generate text with streaming (returns tokens as they are generated)
    pub fn generate_text_streaming<F>(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        mut callback: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&str) -> Result<bool>, // Returns false to stop generation
    {
        // For now, implement as non-streaming but call callback with full result
        // TODO: Implement true streaming in future iterations
        let output = self.generate_text_with_config(prompt, config)?;

        // Call callback with the generated text
        let should_continue = callback(&output.text)?;
        if !should_continue {
            return Ok(GenerationOutput::new(
                output.text,
                output.tokens_generated,
                crate::inference::generation::StopReason::Error("Stopped by callback".to_string()),
            ));
        }

        Ok(output)
    }

    /// Set generation configuration
    pub fn set_generation_config(&mut self, config: GenerationConfig) {
        self.default_config = config;
    }

    /// Get current generation configuration
    pub fn generation_config(&self) -> &GenerationConfig {
        &self.default_config
    }

    /// Clear generation cache
    pub fn clear_cache(&mut self) {
        self.generation_cache.clear();
    }

    /// Get tokenizer reference
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Generate text with reasoning awareness
    pub fn generate_with_reasoning(&mut self, prompt: &str) -> Result<ReasoningOutput> {
        self.text_generator.generate_with_reasoning(
            &mut self.model,
            &self.tokenizer,
            prompt,
            &self.default_config,
        )
    }

    /// Generate text with reasoning awareness and custom config
    pub fn generate_with_reasoning_config(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<ReasoningOutput> {
        self.text_generator.generate_with_reasoning(
            &mut self.model,
            &self.tokenizer,
            prompt,
            config,
        )
    }

    /// Generate text with automatic reasoning detection
    pub fn generate_with_reasoning_detection(
        &mut self,
        prompt: &str,
    ) -> Result<(GenerationOutput, Option<ReasoningOutput>)> {
        self.text_generator.generate_with_reasoning_detection(
            &mut self.model,
            &self.tokenizer,
            prompt,
            &self.default_config,
        )
    }

    /// Generate structured reasoning for a given prompt
    pub fn generate_structured_reasoning(&mut self, prompt: &str) -> Result<ReasoningOutput> {
        self.text_generator.generate_structured_reasoning(
            &mut self.model,
            &self.tokenizer,
            prompt,
            &self.default_config,
        )
    }

    /// Solve a mathematical problem with reasoning
    pub fn solve_math_problem(&mut self, problem: &str) -> Result<ReasoningOutput> {
        let math_prompt = format!(
            "Solve this mathematical problem step by step: {}\n\n<think>Let me break this down step by step and show my reasoning.</think>",
            problem
        );

        let mut config = self.default_config.clone();
        #[cfg(test)]
        {
            config.max_tokens = config.max_tokens.min(8);
        }
        #[cfg(not(test))]
        {
            config.max_tokens = 512;
        }

        self.generate_with_reasoning_config(&math_prompt, &config)
    }

    /// Solve a mathematical problem with detailed step-by-step reasoning
    pub fn solve_math_problem_detailed(&mut self, problem: &str) -> Result<MathSolutionOutput> {
        let reasoning_output = self.solve_math_problem(problem)?;

        // Extract final answer using simple string matching
        let final_answer = self.extract_final_answer(&reasoning_output.final_answer);

        Ok(MathSolutionOutput {
            problem: problem.to_string(),
            reasoning_steps: reasoning_output.thinking_chain,
            final_answer,
            confidence: reasoning_output.confidence,
        })
    }

    /// Explain code with reasoning
    pub fn explain_code(&mut self, code: &str) -> Result<ReasoningOutput> {
        let code_prompt = format!(
            "Explain this code step by step:\n\n```\n{}\n```\n\n<think>Let me analyze this code line by line and explain what it does.</think>",
            code
        );

        let mut config = self.default_config.clone();
        #[cfg(test)]
        {
            config.max_tokens = config.max_tokens.min(8);
        }
        #[cfg(not(test))]
        {
            config.max_tokens = 512; // Allow more tokens for detailed explanations
        }

        self.generate_with_reasoning_config(&code_prompt, &config)
    }

    /// Explain code with detailed analysis
    pub fn explain_code_detailed(
        &mut self,
        code: &str,
        language: Option<&str>,
    ) -> Result<CodeExplanationOutput> {
        let language_hint = language
            .map(|lang| format!(" ({})", lang))
            .unwrap_or_default();
        let code_prompt = format!(
            "Analyze and explain this{} code in detail:\n\n```\n{}\n```\n\n<think>Let me break down this code step by step, explaining the purpose, logic, and any important details.</think>",
            language_hint, code
        );

        let config = self.default_config.clone();
        let reasoning_output = self.generate_with_reasoning_config(&code_prompt, &config)?;

        // Extract code summary from final answer
        let summary = self.extract_code_summary(&reasoning_output.final_answer);

        Ok(CodeExplanationOutput {
            original_code: code.to_string(),
            language: language.map(|s| s.to_string()),
            reasoning_steps: reasoning_output.thinking_chain,
            summary,
            confidence: reasoning_output.confidence,
        })
    }

    /// Solve logical reasoning problems
    pub fn solve_logical_problem(&mut self, problem: &str) -> Result<ReasoningOutput> {
        let logic_prompt = format!(
            "Solve this logical reasoning problem step by step: {}\n\n<think>Let me work through this logic problem systematically, considering all the given information and constraints.</think>",
            problem
        );

        let mut config = self.default_config.clone();
        #[cfg(test)]
        {
            config.max_tokens = config.max_tokens.min(8);
        }
        #[cfg(not(test))]
        {
            config.max_tokens = 512;
        }

        self.generate_with_reasoning_config(&logic_prompt, &config)
    }

    /// Solve logical reasoning problems with detailed analysis
    pub fn solve_logical_problem_detailed(
        &mut self,
        problem: &str,
    ) -> Result<LogicalSolutionOutput> {
        let reasoning_output = self.solve_logical_problem(problem)?;

        // Extract logical conclusion from final answer
        let conclusion = self.extract_logical_conclusion(&reasoning_output.final_answer);

        Ok(LogicalSolutionOutput {
            problem: problem.to_string(),
            reasoning_steps: reasoning_output.thinking_chain,
            conclusion,
            confidence: reasoning_output.confidence,
        })
    }

    /// General problem solving with adaptive prompting
    pub fn solve_problem(
        &mut self,
        problem: &str,
        problem_type: ProblemType,
    ) -> Result<ReasoningOutput> {
        match problem_type {
            ProblemType::Mathematical => self.solve_math_problem(problem),
            ProblemType::Logical => self.solve_logical_problem(problem),
            ProblemType::CodeAnalysis => self.explain_code(problem),
            ProblemType::General => {
                let general_prompt = format!(
                    "Analyze and solve this problem: {}\n\n<think>Let me think about this problem carefully and work through it step by step.</think>",
                    problem
                );
                let config = self.default_config.clone();
                self.generate_with_reasoning_config(&general_prompt, &config)
            }
        }
    }

    // Helper methods for extracting structured information from reasoning

    /// Extract final numerical answer from text
    fn extract_final_answer(&self, text: &str) -> Option<String> {
        let lower_text = text.to_lowercase();

        // Look for common answer patterns
        if let Some(pos) = lower_text.find("answer is ") {
            let after_answer = &text[pos + 10..];
            if let Some(number) = self.extract_first_number(after_answer) {
                return Some(number);
            }
        }

        if let Some(pos) = lower_text.find("result: ") {
            let after_result = &text[pos + 8..];
            if let Some(number) = self.extract_first_number(after_result) {
                return Some(number);
            }
        }

        // Look for equals sign
        if let Some(pos) = text.rfind('=') {
            let after_equals = &text[pos + 1..];
            if let Some(number) = self.extract_first_number(after_equals) {
                return Some(number);
            }
        }

        None
    }

    /// Extract the first number from a string
    fn extract_first_number(&self, text: &str) -> Option<String> {
        let mut number_str = String::new();
        let mut found_digit = false;

        for ch in text.trim().chars() {
            if ch.is_ascii_digit() || (ch == '.' && found_digit && !number_str.contains('.')) {
                number_str.push(ch);
                found_digit = true;
            } else if found_digit || !ch.is_whitespace() {
                break;
            }
        }

        if found_digit && !number_str.is_empty() {
            Some(number_str)
        } else {
            None
        }
    }

    /// Extract code summary from final answer
    fn extract_code_summary(&self, text: &str) -> String {
        // Extract the first sentence or paragraph as summary
        if let Some(period_pos) = text.find('.') {
            text[..period_pos + 1].trim().to_string()
        } else {
            text.trim().to_string()
        }
    }

    /// Extract logical conclusion from final answer
    fn extract_logical_conclusion(&self, text: &str) -> String {
        // Look for conclusion indicators
        if let Some(therefore_pos) = text.to_lowercase().find("therefore") {
            text[therefore_pos..].trim().to_string()
        } else if let Some(conclusion_pos) = text.to_lowercase().find("conclusion") {
            text[conclusion_pos..].trim().to_string()
        } else {
            text.trim().to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeepSeekR1Model, ModelConfig};

    #[test]
    fn test_inference_engine_creation() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let engine = InferenceEngine::new(model);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_inference_engine_with_configs() {
        let model_config = ModelConfig::default();
        let model = DeepSeekR1Model::new(model_config).unwrap();

        let tokenizer_config = TokenizerConfig::default();
        let sampling_config = SamplingConfig::default();
        let generation_config = GenerationConfig::default();

        let engine = InferenceEngine::with_configs(
            model,
            tokenizer_config,
            sampling_config,
            generation_config,
        );
        assert!(engine.is_ok());
    }

    #[test]
    fn test_generation_config_management() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let mut engine = InferenceEngine::new(model).unwrap();

        // Test default config
        let default_config = engine.generation_config();
        assert_eq!(default_config.max_tokens, 256);

        // Test setting new config
        let mut new_config = GenerationConfig::default();
        new_config.max_tokens = 512;
        engine.set_generation_config(new_config);

        let updated_config = engine.generation_config();
        assert_eq!(updated_config.max_tokens, 512);
    }

    #[test]
    fn test_cache_management() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let mut engine = InferenceEngine::new(model).unwrap();

        // Should not panic
        engine.clear_cache();
    }

    #[test]
    fn test_tokenizer_access() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let engine = InferenceEngine::new(model).unwrap();

        let tokenizer = engine.tokenizer();
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_problem_solving_methods_exist() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let engine = InferenceEngine::new(model).unwrap();

        // Just verify the engine was created successfully - don't run actual inference
        assert!(engine.tokenizer().vocab_size() > 0);
        assert_eq!(engine.generation_config().max_tokens, 256);
    }

    #[test]
    fn test_extract_first_number() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let engine = InferenceEngine::new(model).unwrap();

        assert_eq!(engine.extract_first_number("42"), Some("42".to_string()));
        assert_eq!(
            engine.extract_first_number("3.14"),
            Some("3.14".to_string())
        );
        assert_eq!(
            engine.extract_first_number("  123  "),
            Some("123".to_string())
        );
        assert_eq!(engine.extract_first_number("no numbers here"), None);
    }

    #[test]
    fn test_extract_final_answer() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let engine = InferenceEngine::new(model).unwrap();

        assert_eq!(
            engine.extract_final_answer("The answer is 42"),
            Some("42".to_string())
        );
        assert_eq!(
            engine.extract_final_answer("2 + 2 = 4"),
            Some("4".to_string())
        );
        assert_eq!(
            engine.extract_final_answer("result: 3.14"),
            Some("3.14".to_string())
        );
    }

    // Note: Full generation testing requires a working model implementation
    // which will be available in later tasks
}
