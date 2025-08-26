//! # Reasoning Chain Generation Demo
//!
//! Demonstrates the DeepSeek R1 reasoning capabilities with step-by-step problem solving.

use ds_r1_rs::inference::reasoning::{
    ReasoningChainParser, ReasoningState, TokenReasoningProcessor,
};
use ds_r1_rs::utils::error::Result;
use ds_r1_rs::utils::tokenizer::{Tokenizer, TokenizerConfig};

fn main() -> Result<()> {
    println!("🧠 DeepSeek R1 Reasoning Chain Generation Demo");
    println!("{}", "=".repeat(50));

    // Initialize tokenizer and reasoning components
    let tokenizer_config = TokenizerConfig::default();
    let tokenizer = Tokenizer::new(tokenizer_config)?;
    let think_start_token = tokenizer.think_start_token_id()?;
    let think_end_token = tokenizer.think_end_token_id()?;

    // Demo 1: Mathematical Problem Solving
    println!("\n📊 Demo 1: Mathematical Problem Solving");
    println!("{}", "-".repeat(40));
    demo_mathematical_reasoning(think_start_token, think_end_token)?;

    // Demo 2: Code Explanation with Logical Breakdown
    println!("\n💻 Demo 2: Code Explanation");
    println!("{}", "-".repeat(40));
    demo_code_explanation(think_start_token, think_end_token)?;

    // Demo 3: Logical Reasoning
    println!("\n🔍 Demo 3: Logical Reasoning");
    println!("{}", "-".repeat(40));
    demo_logical_reasoning(think_start_token, think_end_token)?;

    // Demo 4: Token-by-Token Processing
    println!("\n⚡ Demo 4: Real-time Token Processing");
    println!("{}", "-".repeat(40));
    demo_token_processing(think_start_token, think_end_token)?;

    // Demo 5: Reasoning Quality Assessment
    println!("\n📈 Demo 5: Reasoning Quality Assessment");
    println!("{}", "-".repeat(40));
    demo_quality_assessment(think_start_token, think_end_token)?;

    Ok(())
}

/// Demonstrate mathematical problem solving with step-by-step reasoning
fn demo_mathematical_reasoning(think_start_token: u32, think_end_token: u32) -> Result<()> {
    let mut parser = ReasoningChainParser::new(think_start_token, think_end_token);

    let math_problem = r#"
What is the area of a rectangle with length 12 meters and width 8 meters?

<think>
I need to find the area of a rectangle. The formula for the area of a rectangle is:
Area = length × width

Given information:
- Length = 12 meters
- Width = 8 meters

Let me calculate step by step:
Area = 12 × 8 = 96
</think>

<think>
Let me verify this calculation:
12 × 8 = 96 square meters

This makes sense because:
- 12 × 8 = (10 + 2) × 8 = (10 × 8) + (2 × 8) = 80 + 16 = 96
- The units are correct: meters × meters = square meters
</think>

The area of the rectangle is 96 square meters.
"#;

    let result = parser.parse_structured(math_problem)?;

    println!("Problem: Rectangle area calculation");
    println!(
        "Reasoning steps: {}",
        result.analysis.total_thinking_sections
    );
    println!(
        "Quality score: {:.2}",
        result.analysis.reasoning_quality_score
    );
    println!("Has step-by-step: {}", result.analysis.has_step_by_step);
    println!("Has verification: {}", result.analysis.has_verification);

    println!("\nReasoning chain:");
    for (i, step) in result.reasoning_output.thinking_chain.iter().enumerate() {
        println!("  {}. {}", i + 1, step.lines().next().unwrap_or(""));
        if step.lines().count() > 1 {
            println!("     [... {} more lines]", step.lines().count() - 1);
        }
    }

    println!("Final answer: {}", result.reasoning_output.final_answer);

    Ok(())
}

/// Demonstrate code explanation with logical breakdown
fn demo_code_explanation(think_start_token: u32, think_end_token: u32) -> Result<()> {
    let mut parser = ReasoningChainParser::new(think_start_token, think_end_token);

    let code_explanation = r#"
Explain what this Rust function does:

```rust
fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}
```

<think>
I need to analyze this Rust function step by step:

1. Function signature: `fn fibonacci(n: u32) -> u32`
   - Takes an unsigned 32-bit integer as input
   - Returns an unsigned 32-bit integer

2. The function uses pattern matching with `match n`
   - If n is 0, return 0 (base case)
   - If n is 1, return 1 (base case)
   - For any other value, it recursively calls itself

3. The recursive case: `fibonacci(n - 1) + fibonacci(n - 2)`
   - This follows the mathematical definition of Fibonacci sequence
   - Each number is the sum of the two preceding numbers
</think>

<think>
Let me trace through an example to verify my understanding:
- fibonacci(0) = 0 (base case)
- fibonacci(1) = 1 (base case)
- fibonacci(2) = fibonacci(1) + fibonacci(0) = 1 + 0 = 1
- fibonacci(3) = fibonacci(2) + fibonacci(1) = 1 + 1 = 2
- fibonacci(4) = fibonacci(3) + fibonacci(2) = 2 + 1 = 3

This matches the Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13...
</think>

This function implements the Fibonacci sequence using recursion. It calculates the nth Fibonacci number by defining two base cases (0 and 1) and recursively computing larger values as the sum of the two preceding numbers. However, this implementation has exponential time complexity due to repeated calculations.
"#;

    let result = parser.parse_structured(code_explanation)?;

    println!("Problem: Code explanation");
    println!("Reasoning sections found:");
    for (i, section) in result.parsed_sections.iter().enumerate() {
        println!(
            "  {}. Type: {:?}, Words: {}, Confidence: {:.2}",
            i + 1,
            section.section_type,
            section.word_count,
            section.confidence
        );
    }

    println!(
        "\nFinal explanation length: {} words",
        result
            .reasoning_output
            .final_answer
            .split_whitespace()
            .count()
    );

    Ok(())
}

/// Demonstrate logical reasoning
fn demo_logical_reasoning(think_start_token: u32, think_end_token: u32) -> Result<()> {
    let mut parser = ReasoningChainParser::new(think_start_token, think_end_token);

    let logic_problem = r#"
All cats are mammals. Fluffy is a cat. Is Fluffy a mammal?

<think>
This is a logical syllogism. Let me break it down:

Premise 1: All cats are mammals
Premise 2: Fluffy is a cat
Conclusion: ?

Using deductive reasoning:
- If ALL cats are mammals (universal statement)
- And Fluffy is a cat (specific instance)
- Then Fluffy must be a mammal (logical conclusion)

This follows the logical form:
- All A are B
- X is A
- Therefore, X is B
</think>

<think>
Let me double-check this reasoning:
- The first premise establishes a universal rule about cats
- The second premise identifies Fluffy as belonging to the category "cat"
- By the transitive property, Fluffy inherits all properties of cats
- Since all cats are mammals, Fluffy must be a mammal

This is a valid deductive argument with true premises, so the conclusion is necessarily true.
</think>

Yes, Fluffy is a mammal. This follows logically from the given premises through deductive reasoning.
"#;

    let result = parser.parse_structured(logic_problem)?;

    println!("Problem: Logical syllogism");
    println!("Analysis:");
    println!(
        "  - Reasoning steps: {}",
        result.analysis.total_thinking_sections
    );
    println!(
        "  - Total thinking tokens: {}",
        result.analysis.total_thinking_tokens
    );
    println!(
        "  - Average step length: {:.1} words",
        result.analysis.avg_thinking_length
    );
    println!(
        "  - Quality score: {:.2}",
        result.analysis.reasoning_quality_score
    );

    Ok(())
}

/// Demonstrate real-time token processing
fn demo_token_processing(think_start_token: u32, think_end_token: u32) -> Result<()> {
    let mut processor = TokenReasoningProcessor::new(think_start_token, think_end_token);

    println!("Simulating token-by-token generation:");

    // Simulate a sequence of tokens being generated
    let token_sequence = vec![
        (1000, "What"),
        (1001, " is"),
        (1002, " 5"),
        (1003, " +"),
        (1004, " 3"),
        (1005, "?"),
        (1006, " "),
        (think_start_token, "<think>"),
        (1007, "I"),
        (1008, " need"),
        (1009, " to"),
        (1010, " add"),
        (1011, " 5"),
        (1012, " and"),
        (1013, " 3"),
        (1014, "."),
        (1015, " 5"),
        (1016, " +"),
        (1017, " 3"),
        (1018, " ="),
        (1019, " 8"),
        (think_end_token, "</think>"),
        (1020, " The"),
        (1021, " answer"),
        (1022, " is"),
        (1023, " 8"),
        (1024, "."),
    ];

    for (token_id, token_text) in token_sequence {
        processor.process_generation_token(token_id, token_text)?;

        match processor.get_state() {
            ReasoningState::Normal => print!("🟢"),
            ReasoningState::Thinking => print!("🧠"),
            ReasoningState::Answering => print!("💬"),
        }
    }

    println!("\n\nFinal reasoning output:");
    let output = processor.get_reasoning_output();
    println!("Thinking: {:?}", output.thinking_chain);
    println!("Answer: {}", output.final_answer);

    Ok(())
}

/// Demonstrate reasoning quality assessment
fn demo_quality_assessment(think_start_token: u32, think_end_token: u32) -> Result<()> {
    let mut parser = ReasoningChainParser::new(think_start_token, think_end_token);

    let examples = vec![
        (
            "High Quality",
            r#"<think>First, I need to understand what the problem is asking. The question wants me to find the derivative of f(x) = x^2. I'll use the power rule: d/dx[x^n] = n*x^(n-1). For x^2, n=2, so the derivative is 2*x^(2-1) = 2x. Let me verify this makes sense by checking the definition of derivative.</think>The derivative is 2x."#,
        ),
        (
            "Medium Quality",
            r#"<think>I need to find the derivative of x^2. Using power rule, it's 2x.</think>The answer is 2x."#,
        ),
        ("Low Quality", r#"<think>derivative</think>2x"#),
    ];

    println!("Comparing reasoning quality across examples:");

    for (label, example) in examples {
        let result = parser.parse_structured(example)?;
        println!("\n{} Example:", label);
        println!(
            "  Quality Score: {:.2}",
            result.analysis.reasoning_quality_score
        );
        println!(
            "  Thinking Tokens: {}",
            result.analysis.total_thinking_tokens
        );
        println!("  Has Step-by-Step: {}", result.analysis.has_step_by_step);
        println!("  Has Verification: {}", result.analysis.has_verification);
    }

    Ok(())
}

/// Helper function to create sample reasoning examples for testing
pub fn create_sample_reasoning_examples() -> Vec<(String, String)> {
    vec![
        (
            "Math Problem".to_string(),
            "What is 15% of 80? <think>To find 15% of 80, I need to multiply 80 by 0.15. Let me calculate: 80 × 0.15 = 80 × (0.1 + 0.05) = 8 + 4 = 12</think> 15% of 80 is 12.".to_string()
        ),
        (
            "Logic Puzzle".to_string(),
            "If all birds can fly and penguins are birds, can penguins fly? <think>This seems like a logical syllogism, but I need to be careful. The premise 'all birds can fly' is actually false in reality. Penguins are birds but cannot fly. This is a case where the premise is incorrect.</think> No, penguins cannot fly, which shows that the premise 'all birds can fly' is false.".to_string()
        ),
        (
            "Programming Concept".to_string(),
            "Explain recursion. <think>Recursion is when a function calls itself. It needs two key components: a base case to stop the recursion, and a recursive case that moves toward the base case. A classic example is calculating factorial: factorial(n) = n * factorial(n-1), with base case factorial(0) = 1.</think> Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem.".to_string()
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_examples_parsing() {
        let examples = create_sample_reasoning_examples();
        let mut parser = ReasoningChainParser::new(100, 101);

        for (title, example) in examples {
            let result = parser.parse(&example);
            assert!(result.is_ok(), "Failed to parse example: {}", title);

            let reasoning_output = result.unwrap();
            assert!(
                !reasoning_output.thinking_chain.is_empty(),
                "No thinking chain found in: {}",
                title
            );
            assert!(
                !reasoning_output.final_answer.is_empty(),
                "No final answer found in: {}",
                title
            );
        }
    }

    #[test]
    fn test_reasoning_quality_metrics() {
        let mut parser = ReasoningChainParser::new(100, 101);

        let high_quality = "<think>First, I'll analyze the problem. Then I'll solve it step by step. Finally, I'll verify my answer.</think>Answer";
        let result = parser.parse_structured(high_quality).unwrap();

        assert!(result.analysis.has_step_by_step);
        assert!(result.analysis.reasoning_quality_score > 0.5);
    }

    #[test]
    fn test_token_processor_state_management() {
        let mut processor = TokenReasoningProcessor::new(100, 101);

        // Test state transitions
        assert_eq!(*processor.get_state(), ReasoningState::Normal);

        processor.process_generation_token(100, "<think>").unwrap();
        assert_eq!(*processor.get_state(), ReasoningState::Thinking);

        processor.process_generation_token(101, "</think>").unwrap();
        assert_eq!(*processor.get_state(), ReasoningState::Answering);
    }
}
