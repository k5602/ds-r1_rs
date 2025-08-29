use ds_r1_rs::model::config::{AttentionType, ModelConfig};
use ds_r1_rs::model::transformer::DeepSeekR1Model;
use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 MLA/MoE Integration Demo");
    println!("===========================\n");

    // Resolve config path (arg0: program, arg1: optional config path)
    let args: Vec<String> = env::args().collect();
    let config_path = if args.len() > 1 {
        args[1].clone()
    } else {
        // Default example config shipped with the repo (relative to crate root)
        "configs/mixed_depth_mla_moe.json".to_string()
    };

    // Load configuration from JSON (fallback to default if missing)
    let config: ModelConfig = if Path::new(&config_path).exists() {
        println!("Loading ModelConfig from: {}", config_path);
        ModelConfig::load_from_file(&config_path)?
    } else {
        eprintln!(
            "Config file not found at '{}'. Falling back to default configuration.",
            config_path
        );
        ModelConfig::default()
    };

    print_config_summary(&config);

    // Build model
    let mut model = DeepSeekR1Model::new(config.clone())?;
    println!("✅ Model initialized\n");

    // Print architecture telemetry (kinds and per-layer stats)
    print_architecture_kinds(&model);
    print_mla_stats(&model);
    print_moe_stats(&model, "Initial");

    // Reset MoE load balancers (if any) to start fresh
    let reset_count = model.reset_all_moe_load_balancers();
    if reset_count > 0 {
        println!("🔄 Reset load balancers for {} MoE layer(s)\n", reset_count);
    }

    // Warm-up: run some forwards to accumulate MoE utilization statistics
    if has_any_moe_layer(&model) {
        println!("🔥 Warming up MoE layers with diverse inputs to gather utilization stats...");
        for i in 0..16 {
            // Keep token IDs well within vocab range
            let a = 1 + i as u32;
            let b = 2 + i as u32;
            let c = 3 + i as u32;
            let _ = model.forward(&[a, b, c])?;
        }
        println!("Warm-up complete.\n");

        // Print updated MoE stats after warm-up
        print_moe_stats(&model, "After warm-up");
    }

    // Note about incremental decoding compatibility
    if matches!(config.attention_type, AttentionType::MLA) || config.mla_every.is_some() {
        println!(
            "⚠️  Note: Incremental decoding (KV cache) is disabled when MLA is active (globally or via mla_every).\n"
        );
    } else {
        println!("ℹ️  Incremental decoding (KV cache) is available with Standard attention.\n");
    }

    println!("Done.");
    Ok(())
}

fn print_config_summary(config: &ModelConfig) {
    println!("Configuration Summary");
    println!("---------------------");
    println!("• Vocabulary size: {}", config.vocab_size);
    println!("• Hidden size: {}", config.hidden_size);
    println!("• Layers: {}", config.num_layers);
    println!("• Heads: {}", config.num_heads);
    println!("• Intermediate size: {}", config.intermediate_size);
    println!("• Max seq len: {}", config.max_seq_len);
    println!("• Attention type: {:?}", config.attention_type);
    println!("• Feed-forward type: {:?}", config.ff_type);
    println!(
        "• MLA every: {}",
        config
            .mla_every
            .map(|n| n.to_string())
            .unwrap_or_else(|| "—".to_string())
    );
    println!(
        "• MoE every: {}",
        config
            .moe_every
            .map(|n| n.to_string())
            .unwrap_or_else(|| "—".to_string())
    );
    println!(
        "• KV compression ratio (MLA): {}",
        config.kv_compression_ratio
    );
    println!("• RoPE theta: {}", config.rope_theta);
    println!("• MoE experts: {}", config.num_experts);
    println!("• MoE experts per token: {}", config.experts_per_token);
    println!();
}

fn print_architecture_kinds(model: &DeepSeekR1Model) {
    println!("Layer Kinds (Attention / FFN)");
    println!("------------------------------");
    let att_kinds = model.layer_attention_kinds();
    let ff_kinds = model.layer_ff_kinds();
    for (idx, (a, f)) in att_kinds.iter().zip(ff_kinds.iter()).enumerate() {
        println!("Layer {:>2}: Attention = {:<8} | FF = {:<5}", idx + 1, a, f);
    }
    println!();
}

fn print_mla_stats(model: &DeepSeekR1Model) {
    let mla_stats = model.layer_mla_compression_stats();
    let has_any = mla_stats.iter().any(|s| s.is_some());
    println!("MLA Compression Stats (per layer)");
    println!("---------------------------------");
    if !has_any {
        println!("(no MLA layers active)");
        println!();
        return;
    }
    for (idx, stat) in mla_stats.iter().enumerate() {
        if let Some((ratio, savings)) = stat {
            println!(
                "Layer {:>2}: compression_ratio = {:.2}, est_kv_memory_savings = {:.1}%",
                idx + 1,
                ratio,
                savings
            );
        }
    }
    println!();
}

fn print_moe_stats(model: &DeepSeekR1Model, label: &str) {
    let moe_utils = model.layer_moe_utilization();
    let moe_losses = model.layer_moe_load_balance_loss();
    let has_any = moe_utils.iter().any(|s| s.is_some());

    println!("MoE Stats — {}", label);
    println!("----------------");
    if !has_any {
        println!("(no MoE layers active)\n");
        return;
    }

    for (idx, (util_opt, loss_opt)) in moe_utils.iter().zip(moe_losses.iter()).enumerate() {
        if let Some(util) = util_opt {
            let loss = loss_opt.unwrap_or(0.0);
            // Basic, compact utilization printout (truncate if very long)
            let preview: String = format!("{:?}", util);
            let preview = if preview.len() > 80 {
                format!("{} ... ({} vals)", &preview[..77], util.len())
            } else {
                preview
            };
            println!(
                "Layer {:>2}: utilization = {}, load_balance_variance = {:.6}",
                idx + 1,
                preview,
                loss
            );
        }
    }
    println!();
}

fn has_any_moe_layer(model: &DeepSeekR1Model) -> bool {
    model.layer_ff_kinds().iter().any(|k| matches!(*k, "MoE"))
}
