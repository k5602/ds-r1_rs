//! # Error Handling
//!
//! Comprehensive error types and handling for the DeepSeek R1 implementation.

use thiserror::Error;

/// Main error type for the DeepSeek R1 implementation
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Forward pass error: {0}")]
    Forward(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Evaluation error: {0}")]
    Evaluation(String),

    #[error("Mathematical operation error: {0}")]
    Math(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, ModelError>;

impl ModelError {
    /// Create a configuration error
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::Config(msg.into())
    }

    /// Create a forward pass error
    pub fn forward<S: Into<String>>(msg: S) -> Self {
        Self::Forward(msg.into())
    }

    /// Create a training error
    pub fn training<S: Into<String>>(msg: S) -> Self {
        Self::Training(msg.into())
    }

    /// Create an inference error
    pub fn inference<S: Into<String>>(msg: S) -> Self {
        Self::Inference(msg.into())
    }

    /// Create a not implemented error
    pub fn not_implemented<S: Into<String>>(msg: S) -> Self {
        Self::NotImplemented(msg.into())
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Config(_) | Self::InvalidInput(_) => false,
            Self::Io(_) | Self::Json(_) => false,
            Self::Tokenization(_)
            | Self::Forward(_)
            | Self::Training(_)
            | Self::Inference(_)
            | Self::Evaluation(_)
            | Self::Math(_) => true,
            Self::NotImplemented(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let config_err = ModelError::config("test config error");
        assert!(matches!(config_err, ModelError::Config(_)));

        let forward_err = ModelError::forward("test forward error");
        assert!(matches!(forward_err, ModelError::Forward(_)));

        let training_err = ModelError::training("test training error");
        assert!(matches!(training_err, ModelError::Training(_)));
    }

    #[test]
    fn test_error_recoverability() {
        let config_err = ModelError::config("test");
        assert!(!config_err.is_recoverable());

        let forward_err = ModelError::forward("test");
        assert!(forward_err.is_recoverable());

        let not_impl_err = ModelError::not_implemented("test");
        assert!(!not_impl_err.is_recoverable());
    }

    #[test]
    fn test_error_display() {
        let err = ModelError::config("test message");
        let display = format!("{}", err);
        assert!(display.contains("Configuration error"));
        assert!(display.contains("test message"));
    }
}
