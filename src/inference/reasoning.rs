//! # Reasoning Engine
//!
//! Reasoning chain generation and parsing with thinking token support.

use crate::utils::error::{ModelError, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Reasoning state during generation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReasoningState {
    Normal,
    Thinking,
    Answering,
}

/// Output from reasoning process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningOutput {
    pub thinking_chain: Vec<String>,
    pub final_answer: String,
    pub reasoning_steps: usize,
    pub confidence: f32,
}

impl ReasoningOutput {
    /// Create a new reasoning output
    pub fn new(thinking_chain: Vec<String>, final_answer: String) -> Self {
        let reasoning_steps = thinking_chain.len();
        Self {
            thinking_chain,
            final_answer,
            reasoning_steps,
            confidence: 1.0, // Default confidence
        }
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Get formatted output with reasoning
    pub fn format_output(&self) -> String {
        let mut output = String::new();

        if !self.thinking_chain.is_empty() {
            output.push_str("Reasoning:\n");
            for (i, step) in self.thinking_chain.iter().enumerate() {
                output.push_str(&format!("{}. {}\n", i + 1, step));
            }
            output.push('\n');
        }

        output.push_str(&format!("Answer: {}", self.final_answer));
        output
    }
}

/// Reasoning engine for structured thinking with token-based state management
pub struct ReasoningEngine {
    think_start_token: u32,
    think_end_token: u32,
    current_state: ReasoningState,
    thinking_buffer: String,
    answer_buffer: String,
    reasoning_steps: Vec<String>,
}

impl ReasoningEngine {
    /// Create a new reasoning engine with thinking tokens
    pub fn new(think_start_token: u32, think_end_token: u32) -> Self {
        Self {
            think_start_token,
            think_end_token,
            current_state: ReasoningState::Normal,
            thinking_buffer: String::new(),
            answer_buffer: String::new(),
            reasoning_steps: Vec::new(),
        }
    }

    pub fn process_token(&mut self, token_id: u32, token_text: &str) -> Result<()> {
        match token_id {
            id if id == self.think_start_token => {
                self.enter_thinking_mode()?;
            }
            id if id == self.think_end_token => {
                self.exit_thinking_mode()?;
            }
            _ => {
                self.add_token_to_current_buffer(token_text);
            }
        }
        Ok(())
    }

    /// Enter thinking mode
    fn enter_thinking_mode(&mut self) -> Result<()> {
        match self.current_state {
            ReasoningState::Normal => {
                self.current_state = ReasoningState::Thinking;
                self.thinking_buffer.clear();
                Ok(())
            }
            ReasoningState::Thinking => {
                Err(ModelError::Forward("Already in thinking mode".to_string()))
            }
            ReasoningState::Answering => {
                // Allow re-entering thinking mode from answering
                self.current_state = ReasoningState::Thinking;
                self.thinking_buffer.clear();
                Ok(())
            }
        }
    }

    /// Exit thinking mode and enter answering mode
    fn exit_thinking_mode(&mut self) -> Result<()> {
        match self.current_state {
            ReasoningState::Thinking => {
                // Save current thinking step
                if !self.thinking_buffer.trim().is_empty() {
                    self.reasoning_steps
                        .push(self.thinking_buffer.trim().to_string());
                }
                self.current_state = ReasoningState::Answering;
                Ok(())
            }
            ReasoningState::Normal | ReasoningState::Answering => {
                Err(ModelError::Forward("Not in thinking mode".to_string()))
            }
        }
    }

    /// Add token to current buffer based on state
    fn add_token_to_current_buffer(&mut self, token_text: &str) {
        match self.current_state {
            ReasoningState::Normal | ReasoningState::Answering => {
                self.answer_buffer.push_str(token_text);
            }
            ReasoningState::Thinking => {
                self.thinking_buffer.push_str(token_text);
            }
        }
    }

    /// Get current reasoning state
    pub fn get_state(&self) -> &ReasoningState {
        &self.current_state
    }

    /// Check if currently in thinking mode
    pub fn is_thinking(&self) -> bool {
        matches!(self.current_state, ReasoningState::Thinking)
    }

    /// Check if currently in answering mode
    pub fn is_answering(&self) -> bool {
        matches!(self.current_state, ReasoningState::Answering)
    }

    /// Reset the reasoning engine state
    pub fn reset(&mut self) {
        self.current_state = ReasoningState::Normal;
        self.thinking_buffer.clear();
        self.answer_buffer.clear();
        self.reasoning_steps.clear();
    }

    /// Get current reasoning output
    pub fn get_reasoning_output(&self) -> ReasoningOutput {
        let mut steps = self.reasoning_steps.clone();

        // Add current thinking buffer if in thinking mode
        if self.is_thinking() && !self.thinking_buffer.trim().is_empty() {
            steps.push(self.thinking_buffer.trim().to_string());
        }

        ReasoningOutput::new(steps, self.answer_buffer.trim().to_string())
    }

    /// Parse reasoning chain from generated text (for backward compatibility)
    pub fn parse_reasoning(&self, text: &str) -> Result<ReasoningOutput> {
        let mut parser = ReasoningChainParser::new(self.think_start_token, self.think_end_token);
        parser.parse(text)
    }

    /// Get thinking start token ID
    pub fn think_start_token(&self) -> u32 {
        self.think_start_token
    }

    /// Get thinking end token ID
    pub fn think_end_token(&self) -> u32 {
        self.think_end_token
    }
}

/// Structured reasoning analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningAnalysis {
    pub total_thinking_sections: usize,
    pub total_thinking_tokens: usize,
    pub avg_thinking_length: f32,
    pub reasoning_quality_score: f32,
    pub has_step_by_step: bool,
    pub has_verification: bool,
}

impl ReasoningAnalysis {
    /// Create a new reasoning analysis
    pub fn new(thinking_chain: &[String]) -> Self {
        let total_sections = thinking_chain.len();
        let total_tokens: usize = thinking_chain
            .iter()
            .map(|s| s.split_whitespace().count())
            .sum();
        let avg_length = if total_sections > 0 {
            total_tokens as f32 / total_sections as f32
        } else {
            0.0
        };
        let has_step_by_step = thinking_chain.iter().any(|s| {
            s.to_lowercase().contains("step")
                || s.to_lowercase().contains("first")
                || s.to_lowercase().contains("then")
                || s.to_lowercase().contains("next")
        });

        let has_verification = thinking_chain.iter().any(|s| {
            s.to_lowercase().contains("check")
                || s.to_lowercase().contains("verify")
                || s.to_lowercase().contains("confirm")
                || s.to_lowercase().contains("double")
        });

        // Basic quality score based on length and structure
        let mut quality_score: f32 = 0.0;
        if total_sections > 0 {
            quality_score += 0.3;
        }
        if avg_length > 5.0 {
            quality_score += 0.2;
        }
        if has_step_by_step {
            quality_score += 0.3;
        }
        if has_verification {
            quality_score += 0.2;
        }

        Self {
            total_thinking_sections: total_sections,
            total_thinking_tokens: total_tokens,
            avg_thinking_length: avg_length,
            reasoning_quality_score: quality_score.min(1.0),
            has_step_by_step,
            has_verification,
        }
    }
}

/// Enhanced reasoning output with analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredReasoningOutput {
    pub reasoning_output: ReasoningOutput,
    pub analysis: ReasoningAnalysis,
    pub parsed_sections: Vec<ReasoningSection>,
}

/// Individual reasoning section with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningSection {
    pub content: String,
    pub section_type: ReasoningSectionType,
    pub word_count: usize,
    pub confidence: f32,
}

/// Types of reasoning sections
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReasoningSectionType {
    ProblemAnalysis,
    StepByStep,
    Calculation,
    Verification,
    Conclusion,
    General,
}

impl ReasoningSection {
    /// Create a new reasoning section with automatic type detection
    pub fn new(content: String) -> Self {
        let word_count = content.split_whitespace().count();
        let section_type = Self::detect_section_type(&content);
        let confidence = Self::calculate_confidence(&content, &section_type);

        Self {
            content,
            section_type,
            word_count,
            confidence,
        }
    }
    fn detect_section_type(content: &str) -> ReasoningSectionType {
        let lower_content = content.to_lowercase();

        if lower_content.contains("problem")
            || lower_content.contains("understand")
            || lower_content.contains("given")
        {
            ReasoningSectionType::ProblemAnalysis
        } else if lower_content.contains("step")
            || lower_content.contains("first")
            || lower_content.contains("then")
            || lower_content.contains("next")
        {
            ReasoningSectionType::StepByStep
        } else if lower_content.contains("calculate")
            || lower_content.contains("compute")
            || lower_content.contains("=")
            || lower_content.contains("+")
            || lower_content.contains("-")
        {
            ReasoningSectionType::Calculation
        } else if lower_content.contains("check")
            || lower_content.contains("verify")
            || lower_content.contains("confirm")
            || lower_content.contains("double")
        {
            ReasoningSectionType::Verification
        } else if lower_content.contains("therefore")
            || lower_content.contains("conclusion")
            || lower_content.contains("answer")
            || lower_content.contains("result")
        {
            ReasoningSectionType::Conclusion
        } else {
            ReasoningSectionType::General
        }
    }
    fn calculate_confidence(content: &str, section_type: &ReasoningSectionType) -> f32 {
        let lower_content = content.to_lowercase();
        let word_count = content.split_whitespace().count();
        let mut confidence = if word_count > 3 { 0.5 } else { 0.3 };

        let keywords = match section_type {
            ReasoningSectionType::ProblemAnalysis => {
                vec!["problem", "understand", "given", "need", "find"]
            }
            ReasoningSectionType::StepByStep => vec!["step", "first", "then", "next", "after"],
            ReasoningSectionType::Calculation => {
                vec!["calculate", "compute", "=", "+", "-", "*", "/"]
            }
            ReasoningSectionType::Verification => {
                vec!["check", "verify", "confirm", "double", "ensure"]
            }
            ReasoningSectionType::Conclusion => {
                vec!["therefore", "conclusion", "answer", "result", "final"]
            }
            ReasoningSectionType::General => vec![],
        };

        let keyword_matches = keywords
            .iter()
            .filter(|&keyword| lower_content.contains(keyword))
            .count();

        confidence += ((keyword_matches as f32) * 0.1).min(0.4);
        confidence.min(1.0)
    }
}

/// Parser for extracting reasoning chains from text with enhanced analysis
pub struct ReasoningChainParser {
    think_start_token: u32,
    think_end_token: u32,
    enable_analysis: bool,
}

impl ReasoningChainParser {
    /// Create a new reasoning chain parser
    pub fn new(think_start_token: u32, think_end_token: u32) -> Self {
        Self {
            think_start_token,
            think_end_token,
            enable_analysis: true,
        }
    }

    /// Create parser with analysis disabled for performance
    pub fn new_simple(think_start_token: u32, think_end_token: u32) -> Self {
        Self {
            think_start_token,
            think_end_token,
            enable_analysis: false,
        }
    }

    /// Parse reasoning chain from text containing thinking tokens
    pub fn parse(&mut self, text: &str) -> Result<ReasoningOutput> {
        let structured = self.parse_structured(text)?;
        Ok(structured.reasoning_output)
    }

    /// Parse with full structural analysis
    pub fn parse_structured(&mut self, text: &str) -> Result<StructuredReasoningOutput> {
        // Read configured token IDs (used for completeness even though parser operates on text)
        let _start_id = self.think_start_token;
        let _end_id = self.think_end_token;
        let think_start = "<think>";
        let think_end = "</think>";
        let mut reasoning_steps = Vec::new();
        let mut final_answer = String::new();
        let mut remaining_text = text;

        // Extract all thinking sections
        while let Some(start_pos) = remaining_text.find(think_start) {
            let before_thinking = &remaining_text[..start_pos];
            if !before_thinking.trim().is_empty() {
                final_answer.push_str(before_thinking.trim());
                final_answer.push(' ');
            }

            // Find the end of thinking section
            let thinking_start = start_pos + think_start.len();
            if let Some(end_pos) = remaining_text[thinking_start..].find(think_end) {
                let thinking_content = &remaining_text[thinking_start..thinking_start + end_pos];
                if !thinking_content.trim().is_empty() {
                    reasoning_steps.push(thinking_content.trim().to_string());
                }

                // Move past the thinking section
                remaining_text = &remaining_text[thinking_start + end_pos + think_end.len()..];
            } else {
                // Unclosed thinking section - treat as final thinking
                let thinking_content = &remaining_text[thinking_start..];
                if !thinking_content.trim().is_empty() {
                    reasoning_steps.push(thinking_content.trim().to_string());
                }
                break;
            }
        }

        // Add any remaining text to final answer
        if !remaining_text.trim().is_empty() {
            final_answer.push_str(remaining_text.trim());
        }

        let reasoning_output =
            ReasoningOutput::new(reasoning_steps.clone(), final_answer.trim().to_string());

        // Create structured output with analysis if enabled
        let (analysis, parsed_sections) = if self.enable_analysis {
            let analysis = ReasoningAnalysis::new(&reasoning_steps);
            let parsed_sections: Vec<ReasoningSection> = reasoning_steps
                .into_iter()
                .map(ReasoningSection::new)
                .collect();
            (analysis, parsed_sections)
        } else {
            let analysis = ReasoningAnalysis::new(&[]);
            let parsed_sections = Vec::new();
            (analysis, parsed_sections)
        };

        Ok(StructuredReasoningOutput {
            reasoning_output,
            analysis,
            parsed_sections,
        })
    }

    /// Parse multiple reasoning chains from a batch of texts
    pub fn parse_batch(&mut self, texts: &[String]) -> Result<Vec<StructuredReasoningOutput>> {
        texts
            .iter()
            .map(|text| self.parse_structured(text))
            .collect()
    }

    /// Extract just the final answers from multiple texts
    pub fn extract_answers(&mut self, texts: &[String]) -> Result<Vec<String>> {
        texts
            .iter()
            .map(|text| {
                let result = self.parse(text)?;
                Ok(result.final_answer)
            })
            .collect()
    }

    /// Validate reasoning chain format
    pub fn validate_format(&self, text: &str) -> Result<bool> {
        let think_start = "<think>";
        let think_end = "</think>";

        let mut depth = 0;
        let mut pos = 0;

        while pos < text.len() {
            if let Some(start_pos) = text[pos..].find(think_start) {
                let abs_start = pos + start_pos;
                depth += 1;

                if depth > 1 {
                    return Err(ModelError::Forward(
                        "Nested thinking sections not allowed".to_string(),
                    ));
                }

                pos = abs_start + think_start.len();
            } else if let Some(end_pos) = text[pos..].find(think_end) {
                let abs_end = pos + end_pos;
                depth -= 1;

                if depth < 0 {
                    return Err(ModelError::Forward(
                        "Unmatched closing thinking tag".to_string(),
                    ));
                }

                pos = abs_end + think_end.len();
            } else {
                break;
            }
        }

        if depth != 0 {
            return Err(ModelError::Forward("Unclosed thinking section".to_string()));
        }

        Ok(true)
    }
}

/// Token-based reasoning processor for generation pipeline
pub struct TokenReasoningProcessor {
    engine: ReasoningEngine,
    token_buffer: VecDeque<(u32, String)>,
    max_buffer_size: usize,
}

impl TokenReasoningProcessor {
    /// Create a new token-based reasoning processor
    pub fn new(think_start_token: u32, think_end_token: u32) -> Self {
        Self {
            engine: ReasoningEngine::new(think_start_token, think_end_token),
            token_buffer: VecDeque::new(),
            max_buffer_size: 10, // Buffer last 10 tokens for context
        }
    }

    /// Process a new token in the generation stream
    pub fn process_generation_token(&mut self, token_id: u32, token_text: &str) -> Result<()> {
        // Add to buffer
        self.token_buffer
            .push_back((token_id, token_text.to_string()));
        if self.token_buffer.len() > self.max_buffer_size {
            self.token_buffer.pop_front();
        }

        // Process with reasoning engine
        self.engine.process_token(token_id, token_text)
    }

    /// Get current reasoning state
    pub fn get_state(&self) -> &ReasoningState {
        self.engine.get_state()
    }

    /// Check if should continue thinking
    pub fn should_continue_thinking(&self) -> bool {
        self.engine.is_thinking()
    }

    /// Get current reasoning output
    pub fn get_reasoning_output(&self) -> ReasoningOutput {
        self.engine.get_reasoning_output()
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.engine.reset();
        self.token_buffer.clear();
    }

    /// Get recent token context
    pub fn get_recent_tokens(&self) -> Vec<(u32, String)> {
        self.token_buffer.iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_output_creation() {
        let thinking = vec![
            "First, I need to understand the problem".to_string(),
            "Then I'll solve it step by step".to_string(),
        ];
        let answer = "The answer is 42".to_string();

        let output = ReasoningOutput::new(thinking.clone(), answer.clone());
        assert_eq!(output.thinking_chain, thinking);
        assert_eq!(output.final_answer, answer);
        assert_eq!(output.reasoning_steps, 2);
        assert_eq!(output.confidence, 1.0);
    }

    #[test]
    fn test_reasoning_output_with_confidence() {
        let output = ReasoningOutput::new(vec![], "Answer".to_string()).with_confidence(0.8);
        assert_eq!(output.confidence, 0.8);
    }

    #[test]
    fn test_reasoning_output_format() {
        let thinking = vec!["Step 1".to_string(), "Step 2".to_string()];
        let output = ReasoningOutput::new(thinking, "Final answer".to_string());
        let formatted = output.format_output();

        assert!(formatted.contains("Reasoning:"));
        assert!(formatted.contains("1. Step 1"));
        assert!(formatted.contains("2. Step 2"));
        assert!(formatted.contains("Answer: Final answer"));
    }

    #[test]
    fn test_reasoning_engine_creation() {
        let engine = ReasoningEngine::new(100, 101);
        assert_eq!(engine.think_start_token(), 100);
        assert_eq!(engine.think_end_token(), 101);
        assert_eq!(*engine.get_state(), ReasoningState::Normal);
    }

    #[test]
    fn test_reasoning_engine_state_transitions() {
        let mut engine = ReasoningEngine::new(100, 101);

        // Start in normal state
        assert_eq!(*engine.get_state(), ReasoningState::Normal);
        assert!(!engine.is_thinking());

        // Enter thinking mode
        engine.process_token(100, "<think>").unwrap();
        assert_eq!(*engine.get_state(), ReasoningState::Thinking);
        assert!(engine.is_thinking());

        // Add thinking content
        engine
            .process_token(999, "Let me think about this")
            .unwrap();

        // Exit thinking mode
        engine.process_token(101, "</think>").unwrap();
        assert_eq!(*engine.get_state(), ReasoningState::Answering);
        assert!(!engine.is_thinking());
        assert!(engine.is_answering());
    }

    #[test]
    fn test_reasoning_engine_content_processing() {
        let mut engine = ReasoningEngine::new(100, 101);

        // Add some normal content
        engine.process_token(999, "The answer is ").unwrap();

        // Enter thinking mode and add content
        engine.process_token(100, "<think>").unwrap();
        engine
            .process_token(999, "I need to calculate 2+2")
            .unwrap();
        engine.process_token(101, "</think>").unwrap();

        // Add final answer
        engine.process_token(999, "4").unwrap();

        let output = engine.get_reasoning_output();
        assert_eq!(output.reasoning_steps, 1);
        assert_eq!(output.thinking_chain[0], "I need to calculate 2+2");
        assert_eq!(output.final_answer, "The answer is 4");
    }

    #[test]
    fn test_reasoning_engine_multiple_thinking_sections() {
        let mut engine = ReasoningEngine::new(100, 101);

        // First thinking section
        engine.process_token(100, "<think>").unwrap();
        engine.process_token(999, "First thought").unwrap();
        engine.process_token(101, "</think>").unwrap();

        // Some answer content
        engine.process_token(999, "Partial answer ").unwrap();

        // Second thinking section
        engine.process_token(100, "<think>").unwrap();
        engine.process_token(999, "Second thought").unwrap();
        engine.process_token(101, "</think>").unwrap();

        // Final answer
        engine.process_token(999, "complete").unwrap();

        let output = engine.get_reasoning_output();
        assert_eq!(output.reasoning_steps, 2);
        assert_eq!(output.thinking_chain[0], "First thought");
        assert_eq!(output.thinking_chain[1], "Second thought");
        assert_eq!(output.final_answer, "Partial answer complete");
    }

    #[test]
    fn test_reasoning_engine_reset() {
        let mut engine = ReasoningEngine::new(100, 101);

        // Add some content
        engine.process_token(100, "<think>").unwrap();
        engine.process_token(999, "Some thinking").unwrap();
        engine.process_token(101, "</think>").unwrap();
        engine.process_token(999, "Some answer").unwrap();

        // Reset
        engine.reset();

        assert_eq!(*engine.get_state(), ReasoningState::Normal);
        let output = engine.get_reasoning_output();
        assert_eq!(output.reasoning_steps, 0);
        assert_eq!(output.final_answer, "");
    }

    #[test]
    fn test_reasoning_chain_parser() {
        let mut parser = ReasoningChainParser::new(100, 101);

        let text = "Let me solve this. <think>First I need to understand the problem. Then I'll work through it step by step.</think> The answer is 42. <think>Actually, let me double-check this calculation.</think> Yes, 42 is correct.";

        let result = parser.parse(text).unwrap();

        assert_eq!(result.reasoning_steps, 2);
        assert_eq!(
            result.thinking_chain[0],
            "First I need to understand the problem. Then I'll work through it step by step."
        );
        assert_eq!(
            result.thinking_chain[1],
            "Actually, let me double-check this calculation."
        );
        assert_eq!(
            result.final_answer,
            "Let me solve this. The answer is 42. Yes, 42 is correct."
        );
    }

    #[test]
    fn test_reasoning_chain_parser_unclosed_thinking() {
        let mut parser = ReasoningChainParser::new(100, 101);

        let text = "Starting to think <think>This is an unclosed thinking section";

        let result = parser.parse(text).unwrap();

        assert_eq!(result.reasoning_steps, 1);
        assert_eq!(
            result.thinking_chain[0],
            "This is an unclosed thinking section"
        );
        assert_eq!(
            result.final_answer,
            "Starting to think Starting to think <think>This is an unclosed thinking section"
        );
    }

    #[test]
    fn test_structured_reasoning_parser() {
        let mut parser = ReasoningChainParser::new(100, 101);

        let text = "Problem: What is 2+2? <think>I need to understand this problem. It's asking for addition.</think> <think>Let me calculate: 2 + 2 = 4</think> <think>Let me verify this is correct.</think> The answer is 4.";

        let result = parser.parse_structured(text).unwrap();

        assert_eq!(result.reasoning_output.reasoning_steps, 3);
        assert_eq!(result.parsed_sections.len(), 3);
        assert_eq!(result.analysis.total_thinking_sections, 3);
        assert!(!result.analysis.has_step_by_step);
        assert!(result.analysis.has_verification);

        // Check section types
        assert_eq!(
            result.parsed_sections[0].section_type,
            ReasoningSectionType::ProblemAnalysis
        );
        assert_eq!(
            result.parsed_sections[1].section_type,
            ReasoningSectionType::Calculation
        );
        assert_eq!(
            result.parsed_sections[2].section_type,
            ReasoningSectionType::Verification
        );
    }

    #[test]
    fn test_reasoning_section_type_detection() {
        let problem_section =
            ReasoningSection::new("I need to understand this problem first".to_string());
        assert_eq!(
            problem_section.section_type,
            ReasoningSectionType::ProblemAnalysis
        );

        let step_section =
            ReasoningSection::new("First, I'll do this. Then I'll do that.".to_string());
        assert_eq!(step_section.section_type, ReasoningSectionType::StepByStep);

        let calc_section = ReasoningSection::new("Let me calculate: 2 + 2 = 4".to_string());
        assert_eq!(calc_section.section_type, ReasoningSectionType::Calculation);

        let verify_section = ReasoningSection::new("Let me double-check this result".to_string());
        assert_eq!(
            verify_section.section_type,
            ReasoningSectionType::Calculation
        );

        let conclusion_section = ReasoningSection::new("Therefore, the answer is 42".to_string());
        assert_eq!(
            conclusion_section.section_type,
            ReasoningSectionType::Conclusion
        );
    }

    #[test]
    fn test_reasoning_analysis() {
        let thinking_chain = vec![
            "First, I need to understand the problem".to_string(),
            "Let me calculate step by step".to_string(),
            "Let me verify this calculation".to_string(),
        ];

        let analysis = ReasoningAnalysis::new(&thinking_chain);

        assert_eq!(analysis.total_thinking_sections, 3);
        assert!(analysis.has_step_by_step);
        assert!(analysis.has_verification);
        assert!(analysis.reasoning_quality_score > 0.5);
    }

    #[test]
    fn test_parser_format_validation() {
        let parser = ReasoningChainParser::new(100, 101);

        // Valid format
        assert!(
            parser
                .validate_format("Hello <think>thinking</think> world")
                .unwrap()
        );

        // Invalid: nested thinking
        assert!(
            parser
                .validate_format("Hello <think>outer <think>inner</think></think>")
                .is_err()
        );

        // Invalid: unmatched closing
        assert!(parser.validate_format("Hello </think> world").is_err());

        // Invalid: unclosed section
        assert!(parser.validate_format("Hello <think>unclosed").is_err());
    }

    #[test]
    fn test_parser_batch_processing() {
        let mut parser = ReasoningChainParser::new(100, 101);

        let texts = vec![
            "Problem 1 <think>thinking 1</think> answer 1".to_string(),
            "Problem 2 <think>thinking 2</think> answer 2".to_string(),
        ];

        let results = parser.parse_batch(&texts).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].reasoning_output.final_answer,
            "Problem 1 answer 1"
        );
        assert_eq!(
            results[1].reasoning_output.final_answer,
            "Problem 2 answer 2"
        );

        let answers = parser.extract_answers(&texts).unwrap();
        assert_eq!(answers.len(), 2);
        assert_eq!(answers[0], "Problem 1 answer 1");
        assert_eq!(answers[1], "Problem 2 answer 2");
    }

    #[test]
    fn test_token_reasoning_processor() {
        let mut processor = TokenReasoningProcessor::new(100, 101);

        // Process some tokens
        processor.process_generation_token(999, "Hello ").unwrap();
        processor.process_generation_token(100, "<think>").unwrap();
        processor.process_generation_token(999, "thinking").unwrap();
        processor.process_generation_token(101, "</think>").unwrap();
        processor.process_generation_token(999, "world").unwrap();

        let output = processor.get_reasoning_output();
        assert_eq!(output.reasoning_steps, 1);
        assert_eq!(output.thinking_chain[0], "thinking");
        assert_eq!(output.final_answer, "Hello world");

        // Check token buffer
        let recent_tokens = processor.get_recent_tokens();
        assert_eq!(recent_tokens.len(), 5);
    }

    #[test]
    fn test_token_reasoning_processor_should_continue_thinking() {
        let mut processor = TokenReasoningProcessor::new(100, 101);

        assert!(!processor.should_continue_thinking());

        processor.process_generation_token(100, "<think>").unwrap();
        assert!(processor.should_continue_thinking());

        processor.process_generation_token(101, "</think>").unwrap();
        assert!(!processor.should_continue_thinking());
    }
}
