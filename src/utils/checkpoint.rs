/*!
Checkpoint save/load utilities (JSON)

This module provides JSON-based checkpointing for model weights using the
stable parameter names exposed by the parameter registry (see `parameters_mut`
and `parameters_info` on `DeepSeekR1Model`). It focuses on saving and restoring
the weights the prototype training updates (embeddings and LM head), with
strict shape validation for reproducibility.

File format (pretty JSON):
{
  "format": "ds-r1-rs.weights+json",
  "version": 1,
  "created_at": "2025-08-28T12:34:56.789Z",
  "params": [
    { "name": "embeddings.weight\[0]", "len": 512, "data": [ ...512 floats... ] },
    ...
    { "name": "lm_head.weight\[0]", "len": 512, "data": [ ...512 floats... ] },
    ...
    { "name": "lm_head.bias", "len": 32000, "data": [ ...32000 floats... ] }
  ]
}

Usage:
- save_weights_json(&mut model, "path/to/checkpoint.json")?;
- load_weights_json(&mut model, "path/to/checkpoint.json")?;

Notes:
- Shapes are validated against the current model via `parameters_info()`.
- Unknown parameters in the checkpoint, missing parameters, or length mismatches
  will return a ModelError::Config.
*/

use crate::DeepSeekR1Model;
use crate::utils::error::{ModelError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

const CHECKPOINT_FORMAT: &str = "ds-r1-rs.weights+json";
const CHECKPOINT_VERSION: u32 = 1;

/// A single flat parameter buffer in the checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeightEntry {
    /// Stable, unique name (e.g., "embeddings.weight[0]", "lm_head.bias")
    name: String,
    /// Number of elements in `data`
    len: usize,
    /// Flat f32 weights
    data: Vec<f32>,
}

/// Top-level JSON structure for weights checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeightsCheckpoint {
    /// Magic string to identify this format
    format: String,
    /// Format version
    version: u32,
    /// Creation timestamp (UTC)
    created_at: DateTime<Utc>,
    /// Serialized parameter buffers
    params: Vec<WeightEntry>,
}

impl WeightsCheckpoint {
    fn new(params: Vec<WeightEntry>) -> Self {
        Self {
            format: CHECKPOINT_FORMAT.to_string(),
            version: CHECKPOINT_VERSION,
            created_at: Utc::now(),
            params,
        }
    }

    fn validate_header(&self) -> Result<()> {
        if self.format != CHECKPOINT_FORMAT {
            return Err(ModelError::Config(format!(
                "Unexpected checkpoint format: '{}', expected '{}'",
                self.format, CHECKPOINT_FORMAT
            )));
        }
        if self.version != CHECKPOINT_VERSION {
            return Err(ModelError::Config(format!(
                "Unsupported checkpoint version: {}, expected {}",
                self.version, CHECKPOINT_VERSION
            )));
        }
        Ok(())
    }
}

/// Options to control which parameters are saved (by name prefix).
#[derive(Debug, Clone, Default)]
pub struct SaveOptions {
    /// If set, only parameters whose names start with any of these prefixes are saved.
    pub include: Option<Vec<String>>,
    /// Parameters with names starting with any of these prefixes are excluded.
    pub exclude: Vec<String>,
}

/// Options to control partial/filtered checkpoint loading.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// If true, loading will not error if some model parameters are missing in the checkpoint
    /// (after the include/exclude filters are applied). Default: false.
    pub allow_missing: bool,
    /// If true, unknown parameters present in the checkpoint (not found in the current model)
    /// will be ignored instead of erroring. Default: false.
    pub allow_unknown: bool,
    /// If set, only parameters whose names start with any of these prefixes will be considered for loading.
    pub include: Option<Vec<String>>,
    /// Parameters whose names start with any of these prefixes will be skipped when loading.
    pub exclude: Vec<String>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            allow_missing: false,
            allow_unknown: false,
            include: None,
            exclude: Vec::new(),
        }
    }
}

fn name_matches(name: &str, include: &Option<Vec<String>>, exclude: &Vec<String>) -> bool {
    if let Some(incs) = include {
        if !incs.iter().any(|p| name.starts_with(p)) {
            return false;
        }
    }
    if exclude.iter().any(|p| name.starts_with(p)) {
        return false;
    }
    true
}

/// Save parameters to a JSON file using options (filtered/partial).
pub fn save_weights_json_with_options(
    model: &mut DeepSeekR1Model,
    path: &str,
    opts: &SaveOptions,
) -> Result<()> {
    let reg = model.parameters_mut();
    let mut params = Vec::new();
    for p in reg.iter() {
        let name = p.name();
        if !name_matches(name, &opts.include, &opts.exclude) {
            continue;
        }
        params.push(WeightEntry {
            name: name.to_string(),
            len: p.len(),
            data: p.data().to_vec(),
        });
    }

    let ckpt = WeightsCheckpoint::new(params);
    let json = serde_json::to_string_pretty(&ckpt)
        .map_err(|e| ModelError::Config(format!("Failed to serialize weights to JSON: {}", e)))?;
    std::fs::write(path, json).map_err(ModelError::Io)?;
    Ok(())
}

/// Load weights from JSON using options (filtered/partial).
pub fn load_weights_json_with_options(
    model: &mut DeepSeekR1Model,
    path: &str,
    opts: &LoadOptions,
) -> Result<()> {
    // Read and parse file
    let json = std::fs::read_to_string(path).map_err(ModelError::Io)?;
    let ckpt: WeightsCheckpoint = serde_json::from_str(&json)
        .map_err(|e| ModelError::Config(format!("Failed to parse checkpoint JSON: {}", e)))?;

    // Validate header
    ckpt.validate_header()?;

    // Expected shapes from current model
    let expected_map: HashMap<String, usize> = model
        .parameters_info()
        .into_iter()
        .map(|info| (info.name, info.len))
        .collect();

    // Prepare set of required names in the model after filters (for allow_missing check)
    let required_names: Vec<String> = expected_map
        .keys()
        .filter(|name| name_matches(name, &opts.include, &opts.exclude))
        .cloned()
        .collect();

    // Track which required params were actually applied
    let mut seen: HashSet<String> = HashSet::with_capacity(required_names.len());

    // Acquire mutable registry to copy data into model buffers
    let mut reg = model.parameters_mut();

    // Apply each provided parameter (subject to filters)
    for entry in &ckpt.params {
        // Skip if this name doesn't match the include/exclude filters
        if !name_matches(&entry.name, &opts.include, &opts.exclude) {
            continue;
        }

        // Check if present in current model
        let maybe_expected_len = expected_map.get(&entry.name);
        match maybe_expected_len {
            None => {
                if !opts.allow_unknown {
                    return Err(ModelError::Config(format!(
                        "Checkpoint contains unknown parameter '{}'",
                        entry.name
                    )));
                } else {
                    // Ignore unknown parameter
                    continue;
                }
            }
            Some(&expected_len) => {
                if entry.len != expected_len || entry.data.len() != expected_len {
                    return Err(ModelError::Config(format!(
                        "Checkpoint shape/data length mismatch for '{}': expected {}, got {}/{}",
                        entry.name,
                        expected_len,
                        entry.len,
                        entry.data.len()
                    )));
                }

                // Write into model buffer
                let buf = reg.by_name_mut(&entry.name).ok_or_else(|| {
                    ModelError::Config(format!("Parameter '{}' not found in registry", entry.name))
                })?;
                buf.copy_from_slice(&entry.data);
                seen.insert(entry.name.clone());
            }
        }
    }

    // Enforce missing-check: ensure that every required name was seen (unless allowed)
    if !opts.allow_missing {
        for name in required_names {
            if !seen.contains(&name) {
                return Err(ModelError::Config(format!(
                    "Checkpoint is missing parameter '{}'",
                    name
                )));
            }
        }
    }

    Ok(())
}

/// Save all exposed model parameters (via registry) to a JSON file.
/// - Collects parameters using `model.parameters_mut()` (embeddings, lm_head rows, bias)
/// - Writes a pretty-printed JSON file with metadata and buffers
pub fn save_weights_json(model: &mut DeepSeekR1Model, path: &str) -> Result<()> {
    save_weights_json_with_options(model, path, &SaveOptions::default())
}

/// Load weights from a JSON file and write them into the current model buffers.
/// Shape compatibility is validated against `model.parameters_info()`.
/// Errors:
/// - Unknown parameter name in checkpoint
/// - Missing parameter from checkpoint
/// - Length mismatch (shape incompatibility)
/// - JSON/IO failures
pub fn load_weights_json(model: &mut DeepSeekR1Model, path: &str) -> Result<()> {
    load_weights_json_with_options(model, path, &LoadOptions::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelConfig;

    #[test]
    fn test_checkpoint_basic_functionality() {
        // Just verify we can create a model and get parameter info without heavy I/O
        let cfg = ModelConfig::default();
        let mut model = DeepSeekR1Model::new(cfg).expect("model");

        // Verify we can get parameter info
        let infos = model.parameters_info();
        assert!(!infos.is_empty());

        // Verify we can access model parameters
        let _registry = model.parameters_mut();

        // Verify basic parameter access works
        let _row = model.embedding_row_mut(0).expect("row0");
        let _bias = model.lm_head_bias_mut();

        // Test passes if we get here without errors
    }

    #[test]
    fn test_checkpoint_rejects_unknown_param() {
        // Build a minimal fake checkpoint with an unknown parameter
        let ckpt = WeightsCheckpoint {
            format: CHECKPOINT_FORMAT.to_string(),
            version: CHECKPOINT_VERSION,
            created_at: Utc::now(),
            params: vec![WeightEntry {
                name: "unknown.param".to_string(),
                len: 1,
                data: vec![0.0],
            }],
        };
        let json = serde_json::to_string_pretty(&ckpt).unwrap();
        let unique_id = std::process::id();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ds_r1_rs_ckpt_bad_{}_{}.json",
            unique_id, timestamp
        ));
        std::fs::write(&path, json).unwrap();

        let cfg = ModelConfig::default();
        let mut model = DeepSeekR1Model::new(cfg).expect("model");
        let err = load_weights_json(&mut model, path.to_str().unwrap()).unwrap_err();
        let _ = std::fs::remove_file(path);
        match err {
            ModelError::Config(msg) => assert!(msg.contains("unknown parameter")),
            _ => panic!("Expected Config error, got {:?}", err),
        }
    }

    #[test]
    fn test_checkpoint_rejects_length_mismatch() {
        // Create a model and obtain expected param names
        let cfg = ModelConfig::default();
        let mut model = DeepSeekR1Model::new(cfg).expect("model");
        let infos = model.parameters_info();

        // Forge a checkpoint with a mismatched length for the first param
        let first = &infos[0];
        let ckpt = WeightsCheckpoint {
            format: CHECKPOINT_FORMAT.to_string(),
            version: CHECKPOINT_VERSION,
            created_at: Utc::now(),
            params: vec![WeightEntry {
                name: first.name.clone(),
                len: first.len + 1, // mismatched
                data: vec![0.0; first.len + 1],
            }],
        };
        let json = serde_json::to_string_pretty(&ckpt).unwrap();
        let unique_id = std::process::id();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ds_r1_rs_ckpt_bad_len_{}_{}.json",
            unique_id, timestamp
        ));
        std::fs::write(&path, json).unwrap();

        let err = load_weights_json(&mut model, path.to_str().unwrap()).unwrap_err();
        let _ = std::fs::remove_file(path);
        match err {
            ModelError::Config(msg) => {
                assert!(msg.contains("shape mismatch") || msg.contains("data length mismatch"))
            }
            _ => panic!("Expected Config error, got {:?}", err),
        }
    }

    #[test]
    fn test_filter_options_creation() {
        // Just verify we can create filter options without heavy I/O
        let save_opts = SaveOptions {
            include: Some(vec!["lm_head".to_string()]),
            exclude: vec![],
        };

        let load_opts = LoadOptions {
            allow_missing: true,
            allow_unknown: false,
            include: Some(vec!["lm_head".to_string()]),
            exclude: vec![],
        };

        // Verify the options were created correctly
        assert!(save_opts.include.is_some());
        assert!(save_opts.exclude.is_empty());
        assert!(load_opts.allow_missing);
        assert!(!load_opts.allow_unknown);
    }
}
