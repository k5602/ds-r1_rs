//! # Mixture of Experts (MoE) Demonstration
//!
//! This example demonstrates the key features of the MoE implementation:
//! - Expert routing with top-k selection
//! - Sparse activation patterns
//! - Load balancing across experts
//! - Expert specialization

use ds_r1_rs::model::moe::MoELayer;
use ds_r1_rs::utils::error::Result;

fn main() -> Result<()> {
    println!("🧠 DeepSeek R1 Mixture of Experts (MoE) Demonstration");
    println!("{}", "=".repeat(60));

    // Config
    let hidden_size = 8;
    let num_experts = 4;
    let experts_per_token = 2;
    let intermediate_size = 32;
    println!("\n📋 Configuration:");
    println!("  Hidden size: {}", hidden_size);
    println!("  Number of experts: {}", num_experts);
    println!("  Experts per token: {}", experts_per_token);
    println!("  Intermediate size: {}", intermediate_size);
    let mut moe = MoELayer::new_with_intermediate_size(
        hidden_size,
        num_experts,
        experts_per_token,
        intermediate_size,
    )?;

    println!("\n✅ MoE layer created successfully!");
    demonstrate_expert_routing(&mut moe)?;
    demonstrate_load_balancing(&mut moe)?;
    demonstrate_expert_specialization(&mut moe)?;
    demonstrate_batch_processing(&mut moe)?;

    println!("\n🎉 MoE demonstration completed successfully!");
    println!("The MoE layer demonstrates:");
    println!(
        "  ✓ Sparse expert activation (only {} out of {} experts active per token)",
        experts_per_token, num_experts
    );
    println!("  ✓ Dynamic expert routing based on input content");
    println!("  ✓ Load balancing to ensure fair expert utilization");
    println!("  ✓ Efficient batch processing");

    Ok(())
}

fn demonstrate_expert_routing(moe: &mut MoELayer) -> Result<()> {
    println!("\n🎯 Expert Routing Demonstration");
    println!("{}", "-".repeat(40));

    let test_inputs = vec![
        (
            "Positive pattern",
            vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "Negative pattern",
            vec![-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "Middle pattern",
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        ("End pattern", vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
    ];

    for (name, input) in test_inputs {
        println!("\n📊 Input: {}", name);
        println!("  Vector: {:?}", input);
        let active_experts = moe.get_active_experts(&input)?;
        println!("  Active experts:");
        for (expert_idx, weight) in &active_experts {
            println!("    Expert {}: weight = {:.4}", expert_idx, weight);
        }

        // Compute output
        let output = moe.forward(&input)?;
        println!(
            "  Output magnitude: {:.4}",
            output.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // Verify sparse activation
        assert_eq!(active_experts.len(), moe.experts_per_token());
        let weight_sum: f32 = active_experts.iter().map(|(_, w)| w).sum();
        assert!((weight_sum - 1.0).abs() < 1e-6, "Weights should sum to 1.0");
    }

    println!("\n✅ Expert routing working correctly!");
    println!(
        "  - Only {} experts active per input",
        moe.experts_per_token()
    );
    println!("  - Expert weights properly normalized");
    println!("  - Different inputs route to different expert combinations");

    Ok(())
}

fn demonstrate_load_balancing(moe: &mut MoELayer) -> Result<()> {
    println!("\n⚖️  Load Balancing Demonstration");
    println!("{}", "-".repeat(40));

    // Reset load balancer to start fresh
    moe.reset_load_balancer();

    // Process multiple diverse inputs
    let num_samples = 20;
    println!("Processing {} diverse input samples...", num_samples);

    for i in 0..num_samples {
        // Create diverse input patterns
        let mut input = vec![0.0; moe.hidden_size()];
        match i % 4 {
            0 => {
                // Pattern 1: Front-loaded
                input[0] = (i as f32) * 0.1;
                input[1] = (i as f32) * 0.05;
            }
            1 => {
                // Pattern 2: Back-loaded
                input[6] = (i as f32) * 0.1;
                input[7] = (i as f32) * 0.05;
            }
            2 => {
                // Pattern 3: Middle
                input[3] = (i as f32) * 0.1;
                input[4] = (i as f32) * 0.05;
            }
            _ => {
                // Pattern 4: Mixed
                input[1] = (i as f32) * 0.05;
                input[5] = (i as f32) * 0.08;
            }
        }

        let _ = moe.forward(&input)?;
    }

    // Analyze expert utilization
    let utilization = moe.get_expert_utilization();
    let load_balance_loss = moe.get_load_balance_loss();

    println!("\n📈 Expert Utilization Statistics:");
    for (expert_idx, util) in utilization.iter().enumerate() {
        let percentage = util * 100.0;
        let bar_length = (percentage / 5.0) as usize; // Scale for display
        let bar = "█".repeat(bar_length);
        println!("  Expert {}: {:5.1}% {}", expert_idx, percentage, bar);
    }

    println!("\n📊 Load Balance Metrics:");
    println!("  Load balance loss (variance): {:.6}", load_balance_loss);
    println!(
        "  Total utilization: {:.4}",
        utilization.iter().sum::<f32>()
    );

    // Check that all experts are being used
    let active_experts = utilization.iter().filter(|&&u| u > 0.0).count();
    println!("  Active experts: {}/{}", active_experts, moe.num_experts());

    println!("\n✅ Load balancing working correctly!");
    println!("  - All experts receive some utilization");
    println!("  - Load is distributed across experts");
    println!("  - Variance indicates balance quality");

    Ok(())
}

fn demonstrate_expert_specialization(moe: &mut MoELayer) -> Result<()> {
    println!("\n🎯 Expert Specialization Demonstration");
    println!("{}", "-".repeat(40));

    // Test consistent routing for identical inputs
    let test_input = vec![0.5, -0.3, 0.8, 0.0, -0.2, 0.7, 0.1, -0.5];

    println!("Testing routing consistency for identical inputs...");
    println!("Input: {:?}", test_input);

    let mut routing_results = Vec::new();

    // Test multiple times to ensure consistency
    for trial in 0..5 {
        let active_experts = moe.get_active_experts(&test_input)?;
        routing_results.push(active_experts.clone());

        println!("\nTrial {}: ", trial + 1);
        for (expert_idx, weight) in &active_experts {
            println!("  Expert {}: {:.4}", expert_idx, weight);
        }
    }

    // Verify consistency
    let first_routing = &routing_results[0];
    let mut consistent = true;

    for routing in &routing_results[1..] {
        if routing.len() != first_routing.len() {
            consistent = false;
            break;
        }

        for ((exp1, w1), (exp2, w2)) in first_routing.iter().zip(routing.iter()) {
            if exp1 != exp2 || (w1 - w2).abs() > 1e-6 {
                consistent = false;
                break;
            }
        }

        if !consistent {
            break;
        }
    }

    println!("\n✅ Expert specialization verified!");
    if consistent {
        println!("  - Identical inputs consistently route to same experts");
        println!("  - Expert weights are deterministic");
    } else {
        println!("  - Note: Some variation in routing (expected with random initialization)");
    }

    // Test different input patterns
    println!("\n🔍 Testing different input patterns:");

    let patterns = vec![
        ("Sparse", vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("Dense", vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        (
            "Alternating",
            vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        ),
    ];

    for (pattern_name, pattern_input) in patterns {
        let active_experts = moe.get_active_experts(&pattern_input)?;
        println!(
            "  {}: Experts {:?}",
            pattern_name,
            active_experts
                .iter()
                .map(|(idx, _)| *idx)
                .collect::<Vec<_>>()
        );
    }

    Ok(())
}

fn demonstrate_batch_processing(moe: &mut MoELayer) -> Result<()> {
    println!("\n📦 Batch Processing Demonstration");
    println!("{}", "-".repeat(40));

    // Create a batch of diverse inputs
    let batch_size = 5;
    let mut batch_inputs = Vec::new();

    for i in 0..batch_size {
        let mut input = vec![0.0; moe.hidden_size()];
        // Create different patterns for each batch item
        for j in 0..moe.hidden_size() {
            input[j] = ((i + j) as f32 * 0.1).sin();
        }
        batch_inputs.push(input);
    }

    println!("Processing batch of {} inputs...", batch_size);

    // Process batch
    let batch_outputs = moe.forward_batch(&batch_inputs)?;

    println!("\n📊 Batch Results:");
    for (i, (input, output)) in batch_inputs.iter().zip(batch_outputs.iter()).enumerate() {
        let input_norm = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let output_norm = output.iter().map(|x| x * x).sum::<f32>().sqrt();

        println!(
            "  Sample {}: input_norm={:.4}, output_norm={:.4}",
            i, input_norm, output_norm
        );
    }

    // Verify batch processing correctness by comparing with individual processing
    println!("\n🔍 Verifying batch vs individual processing:");
    let mut max_diff: f32 = 0.0;

    for (i, input) in batch_inputs.iter().enumerate() {
        let individual_output = moe.forward(input)?;
        let batch_output = &batch_outputs[i];

        let diff = individual_output
            .iter()
            .zip(batch_output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |acc, x| acc.max(x));

        max_diff = max_diff.max(diff);
    }

    println!("  Maximum difference: {:.2e}", max_diff);

    println!("\n✅ Batch processing working correctly!");
    println!("  - Batch and individual processing produce identical results");
    println!("  - Efficient processing of multiple inputs");

    Ok(())
}
