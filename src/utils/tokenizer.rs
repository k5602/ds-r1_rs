//! # Tokenizer
//!
//! Basic tokenizer implementation for text processing.

use crate::utils::error::{ModelError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub pad_token: String,
    pub unk_token: String,
    pub bos_token: String,
    pub eos_token: String,
    pub think_start_token: String,
    pub think_end_token: String,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            pad_token: "<pad>".to_string(),
            unk_token: "<unk>".to_string(),
            bos_token: "<bos>".to_string(),
            eos_token: "<eos>".to_string(),
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
        }
    }
}

/// Basic tokenizer implementation
pub struct Tokenizer {
    config: TokenizerConfig,
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
}

impl Tokenizer {
    /// Create a new tokenizer with basic vocabulary
    pub fn new(config: TokenizerConfig) -> Result<Self> {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        // Add special tokens
        let special_tokens = [&config.pad_token,
            &config.unk_token,
            &config.bos_token,
            &config.eos_token,
            &config.think_start_token,
            &config.think_end_token];

        for (id, token) in special_tokens.iter().enumerate() {
            let token_id = id as u32;
            vocab.insert((*token).clone(), token_id);
            reverse_vocab.insert(token_id, (*token).clone());
        }

        // Add basic vocabulary (simplified for prototype)
        let basic_chars =
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-+=*/()[]{}:;\"'";
        for (i, ch) in basic_chars.chars().enumerate() {
            let token_id = (special_tokens.len() + i) as u32;
            let token = ch.to_string();
            vocab.insert(token.clone(), token_id);
            reverse_vocab.insert(token_id, token);
        }

        Ok(Self {
            config,
            vocab,
            reverse_vocab,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokens = vec![self.bos_token_id()?];

        for ch in text.chars() {
            let token = ch.to_string();
            let token_id = self
                .vocab
                .get(&token)
                .copied()
                .unwrap_or_else(|| self.unk_token_id().unwrap());
            tokens.push(token_id);
        }

        tokens.push(self.eos_token_id()?);
        Ok(tokens)
    }

    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut text = String::new();

        for &token_id in token_ids {
            if let Some(token) = self.reverse_vocab.get(&token_id) {
                // Skip special tokens in output (except thinking tokens)
                if token == &self.config.bos_token
                    || token == &self.config.eos_token
                    || token == &self.config.pad_token
                {
                    continue;
                }
                text.push_str(token);
            } else {
                text.push_str(&self.config.unk_token);
            }
        }

        Ok(text)
    }

    /// Get pad token ID
    pub fn pad_token_id(&self) -> Result<u32> {
        self.vocab
            .get(&self.config.pad_token)
            .copied()
            .ok_or_else(|| ModelError::Tokenization("Pad token not found".to_string()))
    }

    /// Get unknown token ID
    pub fn unk_token_id(&self) -> Result<u32> {
        self.vocab
            .get(&self.config.unk_token)
            .copied()
            .ok_or_else(|| ModelError::Tokenization("Unknown token not found".to_string()))
    }

    /// Get beginning of sequence token ID
    pub fn bos_token_id(&self) -> Result<u32> {
        self.vocab
            .get(&self.config.bos_token)
            .copied()
            .ok_or_else(|| ModelError::Tokenization("BOS token not found".to_string()))
    }

    /// Get end of sequence token ID
    pub fn eos_token_id(&self) -> Result<u32> {
        self.vocab
            .get(&self.config.eos_token)
            .copied()
            .ok_or_else(|| ModelError::Tokenization("EOS token not found".to_string()))
    }

    /// Get thinking start token ID
    pub fn think_start_token_id(&self) -> Result<u32> {
        self.vocab
            .get(&self.config.think_start_token)
            .copied()
            .ok_or_else(|| ModelError::Tokenization("Think start token not found".to_string()))
    }

    /// Get thinking end token ID
    pub fn think_end_token_id(&self) -> Result<u32> {
        self.vocab
            .get(&self.config.think_end_token)
            .copied()
            .ok_or_else(|| ModelError::Tokenization("Think end token not found".to_string()))
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Check if token is a thinking token
    pub fn is_thinking_token(&self, token_id: u32) -> bool {
        if let (Ok(start_id), Ok(end_id)) = (self.think_start_token_id(), self.think_end_token_id())
        {
            token_id == start_id || token_id == end_id
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_creation() {
        let config = TokenizerConfig::default();
        let tokenizer = Tokenizer::new(config);
        assert!(tokenizer.is_ok());
    }

    #[test]
    fn test_encode_decode() {
        let config = TokenizerConfig::default();
        let tokenizer = Tokenizer::new(config).unwrap();

        let text = "Hello world!";
        let tokens = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_special_tokens() {
        let config = TokenizerConfig::default();
        let tokenizer = Tokenizer::new(config).unwrap();

        assert!(tokenizer.pad_token_id().is_ok());
        assert!(tokenizer.unk_token_id().is_ok());
        assert!(tokenizer.bos_token_id().is_ok());
        assert!(tokenizer.eos_token_id().is_ok());
        assert!(tokenizer.think_start_token_id().is_ok());
        assert!(tokenizer.think_end_token_id().is_ok());
    }

    #[test]
    fn test_thinking_tokens() {
        let config = TokenizerConfig::default();
        let tokenizer = Tokenizer::new(config).unwrap();

        let start_id = tokenizer.think_start_token_id().unwrap();
        let end_id = tokenizer.think_end_token_id().unwrap();

        assert!(tokenizer.is_thinking_token(start_id));
        assert!(tokenizer.is_thinking_token(end_id));
        assert!(!tokenizer.is_thinking_token(999));
    }

    #[test]
    fn test_vocab_size() {
        let config = TokenizerConfig::default();
        let tokenizer = Tokenizer::new(config).unwrap();

        assert!(tokenizer.vocab_size() > 0);
    }
}
