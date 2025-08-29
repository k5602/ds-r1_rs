//! # Test Problem Sets
//!
//! Comprehensive test datasets for evaluating reasoning capabilities.

use crate::utils::evaluation::{
    DifficultyLevel, ProblemCategory, ReasoningBenchmark, ReasoningProblem,
};

/// Collection of comprehensive test problem sets
pub struct TestDatasets;

impl TestDatasets {
    /// Create comprehensive mathematical reasoning test set
    pub fn create_comprehensive_math_dataset() -> ReasoningBenchmark {
        ReasoningBenchmark {
            name: "Comprehensive Mathematics".to_string(),
            problems: vec![
                // Basic Arithmetic - Easy
                ReasoningProblem {
                    id: "math_comp_001".to_string(),
                    problem_text: "Calculate: 15 + 27 - 8".to_string(),
                    expected_answer: "34".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "math_comp_002".to_string(),
                    problem_text: "What is 12 × 7?".to_string(),
                    expected_answer: "84".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "math_comp_003".to_string(),
                    problem_text: "Divide 144 by 12".to_string(),
                    expected_answer: "12".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Easy,
                },

                // Percentages and Fractions - Easy to Medium
                ReasoningProblem {
                    id: "math_comp_004".to_string(),
                    problem_text: "What is 25% of 200?".to_string(),
                    expected_answer: "50".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "math_comp_005".to_string(),
                    problem_text: "Convert 3/4 to a decimal".to_string(),
                    expected_answer: "0.75".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "math_comp_006".to_string(),
                    problem_text: "If a shirt costs $40 and is on sale for 30% off, what is the sale price?".to_string(),
                    expected_answer: "$28".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Medium,
                },

                // Algebra - Medium
                ReasoningProblem {
                    id: "math_comp_007".to_string(),
                    problem_text: "Solve for x: 3x + 7 = 22".to_string(),
                    expected_answer: "5".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Medium,
                },
                ReasoningProblem {
                    id: "math_comp_008".to_string(),
                    problem_text: "If y = 2x + 3 and x = 4, what is y?".to_string(),
                    expected_answer: "11".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Medium,
                },
                ReasoningProblem {
                    id: "math_comp_009".to_string(),
                    problem_text: "Solve the system: x + y = 10, x - y = 2".to_string(),
                    expected_answer: "x = 6, y = 4".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Medium,
                },

                // Geometry - Medium
                ReasoningProblem {
                    id: "math_comp_010".to_string(),
                    problem_text: "What is the area of a rectangle with length 8 cm and width 5 cm?".to_string(),
                    expected_answer: "40 square cm".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Medium,
                },
                ReasoningProblem {
                    id: "math_comp_011".to_string(),
                    problem_text: "Find the circumference of a circle with radius 7 cm (use π ≈ 3.14)".to_string(),
                    expected_answer: "43.96 cm".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Medium,
                },

                // Word Problems - Medium to Hard
                ReasoningProblem {
                    id: "math_comp_012".to_string(),
                    problem_text: "A train travels 240 km in 3 hours. What is its average speed?".to_string(),
                    expected_answer: "80 km/h".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Medium,
                },
                ReasoningProblem {
                    id: "math_comp_013".to_string(),
                    problem_text: "Sarah has twice as many apples as John. Together they have 18 apples. How many does each person have?".to_string(),
                    expected_answer: "John has 6, Sarah has 12".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Medium,
                },

                // Advanced Problems - Hard
                ReasoningProblem {
                    id: "math_comp_014".to_string(),
                    problem_text: "Find the roots of the quadratic equation: x² - 5x + 6 = 0".to_string(),
                    expected_answer: "x = 2 or x = 3".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Hard,
                },
                ReasoningProblem {
                    id: "math_comp_015".to_string(),
                    problem_text: "A compound interest investment of $1000 at 5% annual rate for 2 years. What is the final amount?".to_string(),
                    expected_answer: "$1102.50".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Hard,
                },
            ],
        }
    }

    /// Create comprehensive code understanding and explanation test cases
    pub fn create_code_understanding_dataset() -> ReasoningBenchmark {
        ReasoningBenchmark {
            name: "Code Understanding and Explanation".to_string(),
            problems: vec![
                // Basic Code Reading - Easy
                ReasoningProblem {
                    id: "code_001".to_string(),
                    problem_text: "What does this function return?\n```rust\nfn double(x: i32) -> i32 {\n    x * 2\n}\n```".to_string(),
                    expected_answer: "Returns the input number multiplied by 2".to_string(),
                    category: ProblemCategory::Programming,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "code_002".to_string(),
                    problem_text: "What will this code print?\n```rust\nlet x = 5;\nlet y = x + 3;\nprintln!(\"{}\", y);\n```".to_string(),
                    expected_answer: "8".to_string(),
                    category: ProblemCategory::Programming,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "code_003".to_string(),
                    problem_text: "What is the purpose of this loop?\n```rust\nfor i in 0..5 {\n    println!(\"Count: {}\", i);\n}\n```".to_string(),
                    expected_answer: "Prints numbers 0 through 4 with labels".to_string(),
                    category: ProblemCategory::Programming,
                    difficulty: DifficultyLevel::Easy,
                },

                // Control Flow - Medium
                ReasoningProblem {
                    id: "code_004".to_string(),
                    problem_text: "What will this function return for input 10?\n```rust\nfn check_number(n: i32) -> &'static str {\n    if n > 0 {\n        \"positive\"\n    } else if n < 0 {\n        \"negative\"\n    } else {\n        \"zero\"\n    }\n}\n```".to_string(),
                    expected_answer: "positive".to_string(),
                    category: ProblemCategory::Programming,
                    difficulty: DifficultyLevel::Medium,
                },
                ReasoningProblem {
                    id: "code_005".to_string(),
                    problem_text: "What is the final value of sum?\n```rust\nlet mut sum = 0;\nfor i in 1..=4 {\n    sum += i;\n}\n```".to_string(),
                    expected_answer: "10".to_string(),
                    category: ProblemCategory::Programming,
                    difficulty: DifficultyLevel::Medium,
                },

                // Data Structures - Medium
                ReasoningProblem {
                    id: "code_006".to_string(),
                    problem_text: "What will this code output?\n```rust\nlet mut vec = vec![1, 2, 3];\nvec.push(4);\nprintln!(\"{:?}\", vec);\n```".to_string(),
                    expected_answer: "[1, 2, 3, 4]".to_string(),
                    category: ProblemCategory::Programming,
                    difficulty: DifficultyLevel::Medium,
                },
                ReasoningProblem {
                    id: "code_007".to_string(),
                    problem_text: "What does this function do?\n```rust\nfn find_max(numbers: &[i32]) -> Option<i32> {\n    numbers.iter().max().copied()\n}\n```".to_string(),
                    expected_answer: "Finds the maximum value in a slice of integers".to_string(),
                    category: ProblemCategory::Programming,
                    difficulty: DifficultyLevel::Medium,
                },

                // Algorithm Analysis - Hard
                ReasoningProblem {
                    id: "code_008".to_string(),
                    problem_text: "What is the time complexity of this function?\n```rust\nfn linear_search(arr: &[i32], target: i32) -> Option<usize> {\n    for (i, &item) in arr.iter().enumerate() {\n        if item == target {\n            return Some(i);\n        }\n    }\n    None\n}\n```".to_string(),
                    expected_answer: "O(n)".to_string(),
                    category: ProblemCategory::Programming,
                    difficulty: DifficultyLevel::Hard,
                },
                ReasoningProblem {
                    id: "code_009".to_string(),
                    problem_text: "Explain what this recursive function does:\n```rust\nfn factorial(n: u32) -> u32 {\n    if n <= 1 {\n        1\n    } else {\n        n * factorial(n - 1)\n    }\n}\n```".to_string(),
                    expected_answer: "Calculates the factorial of n (n!)".to_string(),
                    category: ProblemCategory::Programming,
                    difficulty: DifficultyLevel::Hard,
                },

                // Error Detection - Hard
                ReasoningProblem {
                    id: "code_010".to_string(),
                    problem_text: "What's wrong with this code?\n```rust\nlet mut vec = vec![1, 2, 3];\nfor item in &vec {\n    if *item == 2 {\n        vec.push(4); // Error here\n    }\n}\n```".to_string(),
                    expected_answer: "Cannot modify vector while borrowing it immutably".to_string(),
                    category: ProblemCategory::Programming,
                    difficulty: DifficultyLevel::Hard,
                },
            ],
        }
    }

    /// Create logical reasoning problems with expected solutions
    pub fn create_logical_reasoning_dataset() -> ReasoningBenchmark {
        ReasoningBenchmark {
            name: "Logical Reasoning Problems".to_string(),
            problems: vec![
                // Basic Logic - Easy
                ReasoningProblem {
                    id: "logic_001".to_string(),
                    problem_text: "All birds can fly. Penguins are birds. Can penguins fly?".to_string(),
                    expected_answer: "No, the premise is incorrect - not all birds can fly".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "logic_002".to_string(),
                    problem_text: "If it's raining, then the ground is wet. The ground is wet. Is it raining?".to_string(),
                    expected_answer: "Not necessarily - the ground could be wet for other reasons".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "logic_003".to_string(),
                    problem_text: "All cats are mammals. Fluffy is a cat. Is Fluffy a mammal?".to_string(),
                    expected_answer: "Yes".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Easy,
                },

                // Conditional Logic - Medium
                ReasoningProblem {
                    id: "logic_004".to_string(),
                    problem_text: "If John studies hard, he will pass the exam. John passed the exam. Did John study hard?".to_string(),
                    expected_answer: "Not necessarily - he could have passed for other reasons".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Medium,
                },
                ReasoningProblem {
                    id: "logic_005".to_string(),
                    problem_text: "Either it will rain or it will be sunny. It's not raining. What can we conclude?".to_string(),
                    expected_answer: "It will be sunny".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Medium,
                },

                // Set Logic - Medium
                ReasoningProblem {
                    id: "logic_006".to_string(),
                    problem_text: "In a class of 30 students, 20 like math, 15 like science, and 10 like both. How many like neither?".to_string(),
                    expected_answer: "5 students".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Medium,
                },

                // Complex Logic - Hard
                ReasoningProblem {
                    id: "logic_007".to_string(),
                    problem_text: "Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a bird. Alice doesn't have the cat. Bob doesn't have the dog. Carol doesn't have the bird. Who has which pet?".to_string(),
                    expected_answer: "Alice has the dog, Bob has the bird, Carol has the cat".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Hard,
                },
                ReasoningProblem {
                    id: "logic_008".to_string(),
                    problem_text: "In a tournament, every team plays every other team exactly once. If there are 6 teams, how many games are played in total?".to_string(),
                    expected_answer: "15 games".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Hard,
                },
            ],
        }
    }

    /// Create ground truth answers and reasoning chains for validation
    pub fn create_reasoning_chain_dataset() -> Vec<ReasoningChainExample> {
        vec![
            ReasoningChainExample {
                problem: "What is 15% of 80?".to_string(),
                expected_reasoning_steps: vec![
                    "I need to calculate 15% of 80".to_string(),
                    "15% can be written as 15/100 or 0.15".to_string(),
                    "So I need to calculate 0.15 × 80".to_string(),
                    "0.15 × 80 = 12".to_string(),
                ],
                expected_answer: "12".to_string(),
                category: ProblemCategory::Mathematics,
            },
            ReasoningChainExample {
                problem: "Solve for x: 2x + 5 = 13".to_string(),
                expected_reasoning_steps: vec![
                    "I have the equation 2x + 5 = 13".to_string(),
                    "First, I'll subtract 5 from both sides: 2x = 13 - 5".to_string(),
                    "This gives me 2x = 8".to_string(),
                    "Now I'll divide both sides by 2: x = 8/2".to_string(),
                    "Therefore x = 4".to_string(),
                ],
                expected_answer: "4".to_string(),
                category: ProblemCategory::Mathematics,
            },
            ReasoningChainExample {
                problem: "All cats are mammals. Fluffy is a cat. Is Fluffy a mammal?".to_string(),
                expected_reasoning_steps: vec![
                    "I'm given that all cats are mammals".to_string(),
                    "I'm also told that Fluffy is a cat".to_string(),
                    "Since Fluffy is a cat, and all cats are mammals".to_string(),
                    "Therefore, Fluffy must be a mammal".to_string(),
                ],
                expected_answer: "Yes".to_string(),
                category: ProblemCategory::Logic,
            },
        ]
    }

    /// Get all comprehensive test datasets
    pub fn get_all_datasets() -> Vec<ReasoningBenchmark> {
        vec![
            Self::create_comprehensive_math_dataset(),
            Self::create_code_understanding_dataset(),
            Self::create_logical_reasoning_dataset(),
        ]
    }
}

/// Example of expected reasoning chain for validation
#[derive(Debug, Clone)]
pub struct ReasoningChainExample {
    pub problem: String,
    pub expected_reasoning_steps: Vec<String>,
    pub expected_answer: String,
    pub category: ProblemCategory,
}

impl ReasoningChainExample {
    /// Validate if a reasoning chain matches the expected pattern
    pub fn validate_reasoning_chain(&self, actual_steps: &[String]) -> f32 {
        if actual_steps.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;
        let expected_count = self.expected_reasoning_steps.len() as f32;

        // Check for key concepts in reasoning steps
        for expected_step in &self.expected_reasoning_steps {
            let expected_lower = expected_step.to_lowercase();
            let expected_words: Vec<&str> = expected_lower.split_whitespace().collect();

            let mut best_match_score: f32 = 0.0;

            for actual_step in actual_steps {
                let actual_lower = actual_step.to_lowercase();
                let actual_words: Vec<&str> = actual_lower.split_whitespace().collect();

                // Check for exact match first
                if expected_lower == actual_lower {
                    best_match_score = 1.0;
                    break;
                }

                // Calculate word overlap
                let overlap = expected_words
                    .iter()
                    .filter(|word| actual_words.contains(word))
                    .count();

                if overlap > 0 {
                    let match_score = (overlap as f32) / (expected_words.len() as f32);
                    best_match_score = best_match_score.max(match_score);
                }
            }

            score += best_match_score;
        }

        (score / expected_count).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_math_dataset() {
        let dataset = TestDatasets::create_comprehensive_math_dataset();
        assert_eq!(dataset.name, "Comprehensive Mathematics");
        assert!(!dataset.problems.is_empty());
        assert!(dataset.problems.len() >= 10);

        // Check that we have problems of different difficulties
        let easy_count = dataset
            .problems
            .iter()
            .filter(|p| p.difficulty == DifficultyLevel::Easy)
            .count();
        let medium_count = dataset
            .problems
            .iter()
            .filter(|p| p.difficulty == DifficultyLevel::Medium)
            .count();
        let hard_count = dataset
            .problems
            .iter()
            .filter(|p| p.difficulty == DifficultyLevel::Hard)
            .count();

        assert!(easy_count > 0);
        assert!(medium_count > 0);
        assert!(hard_count > 0);
    }

    #[test]
    fn test_code_understanding_dataset() {
        let dataset = TestDatasets::create_code_understanding_dataset();
        assert_eq!(dataset.name, "Code Understanding and Explanation");
        assert!(!dataset.problems.is_empty());

        // All problems should be programming category
        assert!(
            dataset
                .problems
                .iter()
                .all(|p| p.category == ProblemCategory::Programming)
        );
    }

    #[test]
    fn test_logical_reasoning_dataset() {
        let dataset = TestDatasets::create_logical_reasoning_dataset();
        assert_eq!(dataset.name, "Logical Reasoning Problems");
        assert!(!dataset.problems.is_empty());

        // All problems should be logic category
        assert!(
            dataset
                .problems
                .iter()
                .all(|p| p.category == ProblemCategory::Logic)
        );
    }

    #[test]
    fn test_reasoning_chain_validation() {
        let examples = TestDatasets::create_reasoning_chain_dataset();
        assert!(!examples.is_empty());

        let example = &examples[0];

        // Test perfect match
        let perfect_score = example.validate_reasoning_chain(&example.expected_reasoning_steps);
        assert!(perfect_score > 0.8);

        // Test partial match
        let partial_steps = vec!["I need to calculate 15%".to_string()];
        let partial_score = example.validate_reasoning_chain(&partial_steps);
        assert!(partial_score > 0.0 && partial_score < perfect_score);

        // Test no match
        let no_match_steps = vec!["Completely unrelated content".to_string()];
        let no_score = example.validate_reasoning_chain(&no_match_steps);
        assert_eq!(no_score, 0.0);
    }

    #[test]
    fn test_get_all_datasets() {
        let datasets = TestDatasets::get_all_datasets();
        assert_eq!(datasets.len(), 3);

        let names: Vec<&str> = datasets.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Comprehensive Mathematics"));
        assert!(names.contains(&"Code Understanding and Explanation"));
        assert!(names.contains(&"Logical Reasoning Problems"));
    }
}
