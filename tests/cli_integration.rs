use assert_cmd::cargo::CommandCargoExt;
use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

/// Create a command for the ds-r1-rs binary.
fn bin() -> Command {
    Command::cargo_bin("ds-r1-rs").expect("binary `ds-r1-rs` not found; ensure it builds")
}

/// Generate a unique temp file path inside OS temp dir.
fn unique_temp_file(name_hint: &str) -> PathBuf {
    let mut p = env::temp_dir();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    p.push(format!("{}_{}_{}.json", name_hint, pid, ts));
    p
}

#[test]
fn cli_config_smoke() {
    let mut cmd = bin();
    cmd.arg("config");
    cmd.timeout(Duration::from_secs(15))
        .assert()
        .success()
        .stdout(predicate::str::contains("Default Model Configuration"))
        .stdout(predicate::str::contains("Vocabulary size"))
        .stdout(predicate::str::contains("Attention type"))
        .stdout(predicate::str::contains("Feed-forward type"));
}

#[test]
fn cli_version_smoke() {
    let mut cmd = bin();
    cmd.arg("version");
    cmd.timeout(Duration::from_secs(10))
        .assert()
        .success()
        .stdout(predicate::str::contains("DeepSeek R1 Rust Implementation"))
        .stdout(predicate::str::contains("Version:"));
}

#[test]
fn cli_test_command_smoke() {
    let mut cmd = bin();
    cmd.arg("test");
    cmd.timeout(Duration::from_secs(60))
        .assert()
        .success()
        .stdout(
            predicate::str::contains("All basic functionality tests passed")
                .or(predicate::str::contains("The project foundation is ready")),
        );
}

/// Avoid heavy generation by asserting usage output when no prompt is provided.
#[test]
fn cli_generate_usage() {
    let mut cmd = bin();
    cmd.arg("generate");
    cmd.timeout(Duration::from_secs(5))
        .assert()
        .success()
        .stdout(
            predicate::str::contains("Usage:").and(predicate::str::contains("generate <prompt>")),
        );
}

/// The `eval` command may be heavy; keep it as ignored to run manually when needed.
#[test]
#[ignore]
fn cli_eval_smoke_heavy() {
    let mut cmd = bin();
    // JSON mode provides structured output; still executes the full harness.
    cmd.args(["eval", "--json"]);
    cmd.timeout(Duration::from_secs(600))
        .assert()
        .success()
        .stdout(predicate::str::contains("["))
        .stdout(predicate::str::contains("benchmark_name"));
}

/// Train for a single step to keep runtime reasonable; still may be heavier than unit tests.
#[test]
#[ignore]
fn cli_train_single_step() {
    let mut cmd = bin();
    cmd.args(["train", "--steps", "1"]);
    cmd.timeout(Duration::from_secs(120))
        .assert()
        .success()
        .stdout(predicate::str::contains("Training for"))
        .stdout(predicate::str::contains("step"));
}

#[test]
fn cli_tokenize_and_detokenize() {
    // Tokenize a short string
    let mut tok = bin();
    tok.args(["tokenize", "Hello", "<think>plan</think>"]);
    let tokenize_out = tok
        .timeout(Duration::from_secs(10))
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let ids_str = String::from_utf8_lossy(&tokenize_out).trim().to_string();
    assert!(
        !ids_str.is_empty(),
        "tokenize produced empty output for a non-empty input"
    );
    let mut detok = bin();
    let mut args = vec!["detokenize".to_string()];
    args.extend(ids_str.split_whitespace().map(|s| s.to_string()));
    detok.args(args);
    detok.timeout(Duration::from_secs(10)).assert().success();
}

/// Save and load a small demo-size checkpoint and perform deterministic generation.
#[test]
fn cli_checkpoint_save_and_load_demo_small() {
    // Save (demo-small to keep artifact under control)
    let save_path = unique_temp_file("ckpt_demo_small");
    let save_path_str = save_path.to_string_lossy().to_string();

    let mut save_cmd = bin();
    save_cmd.args(["save-weights", &save_path_str, "--demo-small"]);
    save_cmd
        .timeout(Duration::from_secs(60))
        .assert()
        .success()
        .stdout(predicate::str::contains("Saved weights"));

    // Load and generate deterministically with a tiny prompt (still demo-small)
    let mut load_cmd = bin();
    load_cmd.args(["load-weights", &save_path_str, "--demo-small", "Hi"]);
    load_cmd
        .timeout(Duration::from_secs(120))
        .assert()
        .success()
        .stdout(predicate::str::contains("Loaded weights"))
        // The deterministic generation path prints this label:
        .stdout(
            predicate::str::contains("Generated (deterministic)")
                .or(predicate::str::contains("Provide a prompt")),
        );

    // Cleanup best-effort
    let _ = std::fs::remove_file(save_path);
}

/// Ensure detokenize handles the usage example values without error.
#[test]
fn cli_detokenize_usage_example() {
    let mut cmd = bin();
    // Example from CLI help; should not error even if decoded text content varies.
    cmd.args(["detokenize", "2", "262", "267", "3"]);
    cmd.timeout(Duration::from_secs(10)).assert().success();
}
