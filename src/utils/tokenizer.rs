#![allow(clippy::needless_return)]
//!
//! Productionized tokenizer wrapper using `tiktoken-rs` encoders with:
//! - Deterministic special token IDs (pad/unk/bos/eos/<think>/</think>)
//! - Round-trip decode via BPE piece mapping plus byte-fallback tokens
//! - Configurable vocabulary cap to match model `vocab_size`
//!
//! Design:
//! - i used a real BPE (cl100k_base by default) from `tiktoken-rs` to segment text.
//! - To keep `vocab_size` consistent with the model, i reserve:
//!   [0..5]  => 6 fixed special tokens
//!   [6..262)=> 256 byte tokens (<byte:0>.. <byte:255>), used as a reliable fallback
//!   [262..vocab_size) => a contiguous mapping of the first N BPE ranks
//!     where N = vocab_size - 262. Any BPE rank >= N is decoded to bytes and represented
//!     using byte tokens, ensuring encode/decode round-trip.
//!
//! Round-trip guarantee for ASCII subset:
//! - ASCII characters always round-trip because they are representable via byte tokens,
//!   even when a BPE piece is out-of-range for the configured `vocab_size`.
//!
//! Notes:
//! - i prepend BOS and append EOS in `encode()` to match the previous prototype behavior.
//! - `decode()` omits BOS/EOS/PAD from output text but preserves <think> and </think> literals.
//! - `encode()` recognizes custom specials literally, so they can be embedded in input text.

use crate::utils::error::{ModelError, Result};
use serde::{Deserialize, Serialize};

use tiktoken_rs::{CoreBPE, cl100k_base, o200k_base};

/// Minimum reserved space: 6 specials + 256 byte tokens
const RESERVED_SPECIAL: usize = 6;
const BYTE_TOKENS: usize = 256;
const BPE_BASE_START: usize = RESERVED_SPECIAL + BYTE_TOKENS; // 262

/// Which tiktoken encoder to use
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TiktokenEncoding {
    Cl100kBase,
    O200kBase,
}

impl Default for TiktokenEncoding {
    fn default() -> Self {
        TiktokenEncoding::Cl100kBase
    }
}

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
    pub encoding: TiktokenEncoding,
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
            encoding: TiktokenEncoding::Cl100kBase,
        }
    }
}

impl TokenizerConfig {
    fn validate(&self) -> Result<()> {
        // Need enough room for specials + byte tokens + at least 1 BPE token
        let min_needed = RESERVED_SPECIAL + BYTE_TOKENS + 1;
        if self.vocab_size < min_needed {
            return Err(ModelError::Tokenization(format!(
                "vocab_size={} too small. Minimum required is {} (6 specials + 256 byte tokens + >=1 BPE)",
                self.vocab_size, min_needed
            )));
        }
        Ok(())
    }
}

/// Backed tokenizer
pub struct Tokenizer {
    config: TokenizerConfig,
    bpe: CoreBPE,

    // Fixed IDs (stable and deterministic)
    pad_id: u32,
    unk_id: u32,
    bos_id: u32,
    eos_id: u32,
    think_start_id: u32,
    think_end_id: u32,

    // Ranges
    byte_min_id: u32,      // inclusive
    byte_max_id: u32,      // inclusive
    bpe_base_id: u32,      // first mapped BPE ID
    bpe_mapped_count: u32, // how many BPE ranks are mapped into vocab
}

impl Tokenizer {
    /// Create a new tokenizer with a tiktoken encoder and constrained vocab mapping
    pub fn new(config: TokenizerConfig) -> Result<Self> {
        config.validate()?;

        // Build BPE
        let bpe = match config.encoding {
            TiktokenEncoding::Cl100kBase => cl100k_base()
                .map_err(|e| ModelError::Tokenization(format!("cl100k_base init failed: {e}")))?,
            TiktokenEncoding::O200kBase => o200k_base()
                .map_err(|e| ModelError::Tokenization(format!("o200k_base init failed: {e}")))?,
        };

        // IDs layout
        let pad_id = 0u32;
        let unk_id = 1u32;
        let bos_id = 2u32;
        let eos_id = 3u32;
        let think_start_id = 4u32;
        let think_end_id = 5u32;

        let byte_min_id = RESERVED_SPECIAL as u32; // 6
        let byte_max_id = (RESERVED_SPECIAL + BYTE_TOKENS - 1) as u32; // 6 + 256 - 1 = 261

        let bpe_base_id = BPE_BASE_START as u32; // 262

        // Max number of BPE ranks that fit in vocab
        let bpe_room = config.vocab_size as isize - BPE_BASE_START as isize;
        if bpe_room <= 0 {
            return Err(ModelError::Tokenization(format!(
                "Insufficient room for BPE mapping: vocab_size={} leaves no space beyond specials+bytes",
                config.vocab_size
            )));
        }
        let bpe_mapped_count = bpe_room as u32;

        Ok(Self {
            config,
            bpe,
            pad_id,
            unk_id,
            bos_id,
            eos_id,
            think_start_id,
            think_end_id,
            byte_min_id,
            byte_max_id,
            bpe_base_id,
            bpe_mapped_count,
        })
    }

    /// Encode text to token IDs (prepends BOS and appends EOS).
    /// - Recognizes custom specials (<bos>, <eos>, <pad>, <unk>, <think>, </think>) literally.
    /// - Uses BPE piece IDs when rank fits into mapped range.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out: Vec<u32> = Vec::new();
        out.push(self.bos_id);

        for seg in self.split_by_custom_specials(text) {
            match seg {
                Segment::Special(s) => {
                    // Map literal to ID
                    if s == self.config.pad_token {
                        out.push(self.pad_id);
                    } else if s == self.config.unk_token {
                        out.push(self.unk_id);
                    } else if s == self.config.bos_token {
                        out.push(self.bos_id);
                    } else if s == self.config.eos_token {
                        out.push(self.eos_id);
                    } else if s == self.config.think_start_token {
                        out.push(self.think_start_id);
                    } else if s == self.config.think_end_token {
                        out.push(self.think_end_id);
                    } else {
                        // Shouldn't happen; treat as plain text fallback
                        self.encode_text_into(&s, &mut out)?;
                    }
                }
                Segment::Text(t) => {
                    self.encode_text_into(&t, &mut out)?;
                }
            }
        }

        out.push(self.eos_id);
        Ok(out)
    }

    /// Decode token IDs to text.
    /// - Skips PAD/BOS/EOS in output.
    /// - Preserves <think> and </think> literals.
    /// - Reassembles consecutive byte tokens into UTF-8.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut text = String::new();
        let mut pending_bytes: Vec<u8> = Vec::new();

        let flush_bytes = |bytes: &mut Vec<u8>, dst: &mut String| {
            if !bytes.is_empty() {
                let s = String::from_utf8_lossy(bytes);
                dst.push_str(&s);
                bytes.clear();
            }
        };

        for &id in token_ids {
            // Skip non-text specials
            if id == self.pad_id || id == self.bos_id || id == self.eos_id {
                continue;
            }

            // Thinking tokens are preserved literally
            if id == self.think_start_id {
                flush_bytes(&mut pending_bytes, &mut text);
                text.push_str(&self.config.think_start_token);
                continue;
            }
            if id == self.think_end_id {
                flush_bytes(&mut pending_bytes, &mut text);
                text.push_str(&self.config.think_end_token);
                continue;
            }

            if self.is_byte_token(id) {
                let b = (id - self.byte_min_id) as u8;
                pending_bytes.push(b);
                continue;
            }

            // Mapped BPE range
            if self.is_bpe_token(id) {
                flush_bytes(&mut pending_bytes, &mut text);

                let rank = id - self.bpe_base_id;
                let piece = self
                    .bpe
                    .decode(vec![rank])
                    .map_err(|e| ModelError::Tokenization(format!("BPE decode failed: {e}")))?;
                text.push_str(&piece);
                continue;
            }

            // Unknown ID: flush bytes and append unk literal
            flush_bytes(&mut pending_bytes, &mut text);
            text.push_str(&self.config.unk_token);
        }

        // Flush any trailing bytes
        if !pending_bytes.is_empty() {
            let s = String::from_utf8_lossy(&pending_bytes);
            text.push_str(&s);
        }

        Ok(text)
    }

    /// Get pad token ID
    pub fn pad_token_id(&self) -> Result<u32> {
        Ok(self.pad_id)
    }

    /// Get unknown token ID
    pub fn unk_token_id(&self) -> Result<u32> {
        Ok(self.unk_id)
    }

    /// Get beginning of sequence token ID
    pub fn bos_token_id(&self) -> Result<u32> {
        Ok(self.bos_id)
    }

    /// Get end of sequence token ID
    pub fn eos_token_id(&self) -> Result<u32> {
        Ok(self.eos_id)
    }

    /// Get thinking start token ID
    pub fn think_start_token_id(&self) -> Result<u32> {
        Ok(self.think_start_id)
    }

    /// Get thinking end token ID
    pub fn think_end_token_id(&self) -> Result<u32> {
        Ok(self.think_end_id)
    }

    /// Vocabulary size exposed to the model and generator
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Check if token is a thinking token
    pub fn is_thinking_token(&self, token_id: u32) -> bool {
        token_id == self.think_start_id || token_id == self.think_end_id
    }

    // ---------- Internal helpers ----------

    fn is_byte_token(&self, id: u32) -> bool {
        id >= self.byte_min_id && id <= self.byte_max_id
    }

    fn is_bpe_token(&self, id: u32) -> bool {
        let last_bpe_id = self.bpe_base_id + self.bpe_mapped_count - 1;
        id >= self.bpe_base_id && id <= last_bpe_id
    }

    /// Encode plain text (no custom specials) into IDs
    fn encode_text_into(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        // Use ordinary encoding to avoid interference from OpenAI’s built-in specials
        let ranks = self.bpe.encode_ordinary(text);

        let mapped_limit = self.bpe_mapped_count; // number of ranks we can map into vocab
        for rank in ranks {
            if rank < mapped_limit {
                out.push(self.bpe_base_id + rank);
            } else {
                // Fallback: decode the piece and emit per-byte tokens
                let piece = self.bpe.decode(vec![rank]).map_err(|e| {
                    ModelError::Tokenization(format!("BPE piece decode failed: {e}"))
                })?;
                self.push_bytes(piece.as_bytes(), out);
            }
        }

        Ok(())
    }

    fn push_bytes(&self, bytes: &[u8], out: &mut Vec<u32>) {
        for &b in bytes {
            out.push(self.byte_min_id + b as u32);
        }
    }

    /// Split input text into alternating text and custom-special segments
    fn split_by_custom_specials(&self, text: &str) -> Vec<Segment> {
        let specials = [
            self.config.pad_token.as_str(),
            self.config.unk_token.as_str(),
            self.config.bos_token.as_str(),
            self.config.eos_token.as_str(),
            self.config.think_start_token.as_str(),
            self.config.think_end_token.as_str(),
        ];

        let mut result = Vec::new();
        let mut cursor = 0usize;
        let text_bytes = text.as_bytes();

        // Find earliest occurrence of any special, iteratively
        while cursor < text.len() {
            let mut next_pos: Option<(usize, &str)> = None;

            for &sp in &specials {
                if sp.is_empty() {
                    continue;
                }
                if let Some(pos) = find_subslice(text_bytes, sp.as_bytes(), cursor) {
                    if next_pos.map_or(true, |(best, _)| pos < best) {
                        next_pos = Some((pos, sp));
                    }
                }
            }

            match next_pos {
                Some((pos, sp)) => {
                    if pos > cursor {
                        // Preceding text
                        let slice = &text[cursor..pos];
                        result.push(Segment::Text(slice.to_string()));
                    }
                    // Special segment
                    result.push(Segment::Special(sp.to_string()));
                    cursor = pos + sp.len();
                }
                None => {
                    // Remaining tail is plain text
                    if cursor < text.len() {
                        result.push(Segment::Text(text[cursor..].to_string()));
                    }
                    break;
                }
            }
        }

        if result.is_empty() {
            result.push(Segment::Text(text.to_string()));
        }

        result
    }
}

/// A segment snippet from input text: either a literal special token or plain text
#[derive(Debug)]
enum Segment {
    Special(String),
    Text(String),
}

/// Find the next occurrence of `needle` in `haystack` starting at `start`.
/// Returns Some(byte_index) or None.
fn find_subslice(haystack: &[u8], needle: &[u8], start: usize) -> Option<usize> {
    if needle.is_empty()
        || haystack.is_empty()
        || start >= haystack.len()
        || needle.len() > haystack.len() - start
    {
        return None;
    }
    // Simple two-pointer scan
    let last = haystack.len() - needle.len();
    let mut i = start;
    while i <= last {
        if &haystack[i..i + needle.len()] == needle {
            return Some(i);
        }
        // Jump heuristic: step by 1 is fine for small inputs
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_creation() {
        let config = TokenizerConfig::default();
        let tok = Tokenizer::new(config);
        assert!(tok.is_ok());
    }

    #[test]
    fn test_encode_decode_ascii_roundtrip() {
        let config = TokenizerConfig::default();
        let tok = Tokenizer::new(config).unwrap();

        let text = "Hello world!";
        let tokens = tok.encode(text).unwrap();
        let decoded = tok.decode(&tokens).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_special_tokens() {
        let config = TokenizerConfig::default();
        let tok = Tokenizer::new(config).unwrap();

        assert!(tok.pad_token_id().is_ok());
        assert!(tok.unk_token_id().is_ok());
        assert!(tok.bos_token_id().is_ok());
        assert!(tok.eos_token_id().is_ok());
        assert!(tok.think_start_token_id().is_ok());
        assert!(tok.think_end_token_id().is_ok());
    }

    #[test]
    fn test_thinking_tokens_detection_and_decode() {
        let config = TokenizerConfig::default();
        let tok = Tokenizer::new(config).unwrap();

        let start_id = tok.think_start_token_id().unwrap();
        let end_id = tok.think_end_token_id().unwrap();
        assert!(tok.is_thinking_token(start_id));
        assert!(tok.is_thinking_token(end_id));
        assert!(!tok.is_thinking_token(999_999));

        // Ensure we preserve <think> and </think> in decode
        let mut ids = vec![tok.bos_token_id().unwrap(), start_id];
        // Encode some text inside thinking
        ids.extend(tok.encode("a+b=c").unwrap());
        ids.push(end_id);
        ids.push(tok.eos_token_id().unwrap());

        let decoded = tok.decode(&ids).unwrap();
        assert!(decoded.contains(&tok.config.think_start_token));
        assert!(decoded.contains(&tok.config.think_end_token));
    }

    #[test]
    fn test_vocab_size() {
        // Also ensure validation enforces minimum size
        let config = TokenizerConfig::default();
        let tok = Tokenizer::new(config).unwrap();
        assert!(tok.vocab_size() >= (RESERVED_SPECIAL + BYTE_TOKENS + 1));
    }

    #[test]
    fn test_split_by_custom_specials() {
        let cfg = TokenizerConfig::default();
        let tok = Tokenizer::new(cfg).unwrap();
        let s = "foo<think>bar</think>baz";
        let segs = tok.split_by_custom_specials(s);
        // Expect: "foo", "<think>", "bar", "</think>", "baz"
        assert_eq!(segs.len(), 5);
        match (&segs[0], &segs[1], &segs[2], &segs[3], &segs[4]) {
            (
                Segment::Text(a),
                Segment::Special(b),
                Segment::Text(c),
                Segment::Special(d),
                Segment::Text(e),
            ) => {
                assert_eq!(a, "foo");
                assert_eq!(b, &tok.config.think_start_token);
                assert_eq!(c, "bar");
                assert_eq!(d, &tok.config.think_end_token);
                assert_eq!(e, "baz");
            }
            _ => panic!("Unexpected segmentation"),
        }
    }

    #[test]
    fn test_byte_fallback_roundtrip() {
        // Construct text with some rare unicode to force possible BPE out-of-range,
        // but our byte-fallback must still round-trip.
        let cfg = TokenizerConfig {
            vocab_size: RESERVED_SPECIAL + BYTE_TOKENS + 64, // small BPE window to trigger byte fallback
            ..TokenizerConfig::default()
        };
        let tok = Tokenizer::new(cfg).unwrap();

        let text = "ASCII ✅ + 漢字 + emoji 🚀";
        let ids = tok.encode(text).unwrap();
        let back = tok.decode(&ids).unwrap();
        assert_eq!(back, text);
    }

    #[test]
    fn test_minimum_vocab_size_validation() {
        // Too small should error
        let bad = TokenizerConfig {
            vocab_size: RESERVED_SPECIAL + BYTE_TOKENS, // no room for any BPE
            ..TokenizerConfig::default()
        };
        let res = Tokenizer::new(bad);
        assert!(res.is_err());
    }

    #[test]
    fn test_bos_eos_wrapping() {
        let tok = Tokenizer::new(TokenizerConfig::default()).unwrap();
        let ids = tok.encode("test").unwrap();
        assert_eq!(ids.first().copied(), tok.bos_token_id().ok());
        assert_eq!(ids.last().copied(), tok.eos_token_id().ok());
    }

    #[test]
    fn test_oov_token_id_decodes_to_unk() {
        let tok = Tokenizer::new(TokenizerConfig::default()).unwrap();
        // Deliberately inject an ID that is not in specials/byte/BPE ranges
        let ids = vec![
            tok.bos_token_id().unwrap(),
            999_999_999,
            tok.eos_token_id().unwrap(),
        ];
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(decoded, tok.config.unk_token);
    }
}
