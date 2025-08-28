//! # Code Examples Dataset
//!
//! Collection of simple algorithms and code snippets with expected explanations
//! for training and demonstrating code understanding capabilities.

use serde::{Deserialize, Serialize};

/// Represents a code example with expected explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    pub name: String,
    pub language: String,
    pub code: String,
    pub description: String,
    pub algorithm_type: AlgorithmType,
    pub difficulty: Difficulty,
    pub expected_explanation: CodeExplanationExpected,
    pub key_concepts: Vec<String>,
}

/// Types of algorithms in the dataset
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlgorithmType {
    Sorting,
    Searching,
    DataStructure,
    Recursion,
    Iteration,
    StringManipulation,
    Mathematical,
    GraphTraversal,
}

/// Difficulty levels for code examples
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Difficulty {
    Beginner,
    Intermediate,
    Advanced,
}

/// Expected explanation structure for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExplanationExpected {
    pub overview: String,
    pub step_by_step: Vec<String>,
    pub complexity_analysis: String,
    pub key_insights: Vec<String>,
    pub best_practices: Vec<String>,
}

/// Dataset containing code examples for training and evaluation
pub struct CodeExamplesDataset {
    examples: Vec<CodeExample>,
}

impl Default for CodeExamplesDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeExamplesDataset {
    /// Create a new dataset with predefined examples
    pub fn new() -> Self {
        let mut dataset = Self {
            examples: Vec::new(),
        };
        
        dataset.populate_examples();
        dataset
    }

    /// Get all examples in the dataset
    pub fn get_examples(&self) -> &[CodeExample] {
        &self.examples
    }

    /// Get examples by algorithm type
    pub fn get_examples_by_type(&self, algorithm_type: AlgorithmType) -> Vec<&CodeExample> {
        self.examples
            .iter()
            .filter(|example| example.algorithm_type == algorithm_type)
            .collect()
    }

    /// Get examples by difficulty
    pub fn get_examples_by_difficulty(&self, difficulty: Difficulty) -> Vec<&CodeExample> {
        self.examples
            .iter()
            .filter(|example| example.difficulty == difficulty)
            .collect()
    }

    /// Get examples by language
    pub fn get_examples_by_language(&self, language: &str) -> Vec<&CodeExample> {
        self.examples
            .iter()
            .filter(|example| example.language == language)
            .collect()
    }

    /// Add a new example to the dataset
    pub fn add_example(&mut self, example: CodeExample) {
        self.examples.push(example);
    }

    /// Get total number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Populate the dataset with predefined examples
    fn populate_examples(&mut self) {
        // Binary Search Algorithm
        self.add_example(CodeExample {
            name: "Binary Search".to_string(),
            language: "rust".to_string(),
            code: r#"fn binary_search(arr: &[i32], target: i32) -> Option<usize> {
    let mut left = 0;
    let mut right = arr.len();
    
    while left < right {
        let mid = left + (right - left) / 2;
        
        if arr[mid] == target {
            return Some(mid);
        } else if arr[mid] < target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    None
}"#.to_string(),
            description: "Efficient search algorithm for sorted arrays".to_string(),
            algorithm_type: AlgorithmType::Searching,
            difficulty: Difficulty::Intermediate,
            expected_explanation: CodeExplanationExpected {
                overview: "Binary search efficiently finds a target value in a sorted array by repeatedly dividing the search space in half.".to_string(),
                step_by_step: vec![
                    "Initialize left and right pointers to define search boundaries".to_string(),
                    "Calculate middle index to avoid integer overflow".to_string(),
                    "Compare middle element with target value".to_string(),
                    "Adjust search boundaries based on comparison result".to_string(),
                    "Repeat until target is found or search space is exhausted".to_string(),
                ],
                complexity_analysis: "Time: O(log n), Space: O(1) - logarithmic search with constant space".to_string(),
                key_insights: vec![
                    "Requires sorted input array for correctness".to_string(),
                    "Divide-and-conquer approach reduces search space exponentially".to_string(),
                    "Overflow-safe middle calculation prevents integer overflow".to_string(),
                ],
                best_practices: vec![
                    "Use inclusive/exclusive bounds consistently".to_string(),
                    "Handle edge cases (empty array, single element)".to_string(),
                    "Return Option type for safe null handling".to_string(),
                ],
            },
            key_concepts: vec!["binary search".to_string(), "divide and conquer".to_string(), "logarithmic complexity".to_string()],
        });

        // Bubble Sort Algorithm
        self.add_example(CodeExample {
            name: "Bubble Sort".to_string(),
            language: "rust".to_string(),
            code: r#"fn bubble_sort(arr: &mut [i32]) {
    let n = arr.len();
    
    for i in 0..n {
        let mut swapped = false;
        
        for j in 0..n - i - 1 {
            if arr[j] > arr[j + 1] {
                arr.swap(j, j + 1);
                swapped = true;
            }
        }
        
        // Early termination if no swaps occurred
        if !swapped {
            break;
        }
    }
}"#.to_string(),
            description: "Simple sorting algorithm that repeatedly steps through the list".to_string(),
            algorithm_type: AlgorithmType::Sorting,
            difficulty: Difficulty::Beginner,
            expected_explanation: CodeExplanationExpected {
                overview: "Bubble sort repeatedly compares adjacent elements and swaps them if they're in wrong order, 'bubbling' larger elements to the end.".to_string(),
                step_by_step: vec![
                    "Outer loop controls the number of passes through the array".to_string(),
                    "Inner loop compares adjacent elements in unsorted portion".to_string(),
                    "Swap elements if they're in wrong order (left > right)".to_string(),
                    "Track if any swaps occurred to enable early termination".to_string(),
                    "Reduce inner loop range as largest elements settle at the end".to_string(),
                ],
                complexity_analysis: "Time: O(n²) worst/average case, O(n) best case, Space: O(1) - quadratic time with constant space".to_string(),
                key_insights: vec![
                    "Each pass guarantees one element reaches its final position".to_string(),
                    "Early termination optimization improves best-case performance".to_string(),
                    "In-place sorting algorithm with stable comparison".to_string(),
                ],
                best_practices: vec![
                    "Use early termination flag for optimization".to_string(),
                    "Reduce comparison range in each iteration".to_string(),
                    "Prefer more efficient algorithms for large datasets".to_string(),
                ],
            },
            key_concepts: vec!["sorting".to_string(), "nested loops".to_string(), "in-place algorithm".to_string()],
        });

        // Fibonacci Sequence (Recursive)
        self.add_example(CodeExample {
            name: "Fibonacci Recursive".to_string(),
            language: "rust".to_string(),
            code: r#"fn fibonacci(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

// Optimized version with memoization
fn fibonacci_memo(n: u32, memo: &mut std::collections::HashMap<u32, u64>) -> u64 {
    if let Some(&result) = memo.get(&n) {
        return result;
    }
    
    let result = match n {
        0 => 0,
        1 => 1,
        _ => fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo),
    };
    
    memo.insert(n, result);
    result
}"#.to_string(),
            description: "Recursive implementation of Fibonacci sequence with memoization optimization".to_string(),
            algorithm_type: AlgorithmType::Recursion,
            difficulty: Difficulty::Intermediate,
            expected_explanation: CodeExplanationExpected {
                overview: "Fibonacci sequence demonstrates recursion with base cases and shows optimization through memoization to avoid redundant calculations.".to_string(),
                step_by_step: vec![
                    "Define base cases for n=0 and n=1 to stop recursion".to_string(),
                    "Recursive case breaks problem into smaller subproblems".to_string(),
                    "Memoization version checks cache before computing".to_string(),
                    "Store computed results to avoid recalculation".to_string(),
                    "Return cached result or compute and cache new result".to_string(),
                ],
                complexity_analysis: "Naive: Time O(2^n), Space O(n). Memoized: Time O(n), Space O(n) - exponential vs linear with memoization".to_string(),
                key_insights: vec![
                    "Naive recursion has exponential time due to overlapping subproblems".to_string(),
                    "Memoization transforms exponential to linear time complexity".to_string(),
                    "Trade-off between space and time complexity".to_string(),
                ],
                best_practices: vec![
                    "Always define clear base cases for recursion".to_string(),
                    "Use memoization for problems with overlapping subproblems".to_string(),
                    "Consider iterative solutions for better space efficiency".to_string(),
                ],
            },
            key_concepts: vec!["recursion".to_string(), "memoization".to_string(), "dynamic programming".to_string()],
        });

        // Linear Search
        self.add_example(CodeExample {
            name: "Linear Search".to_string(),
            language: "rust".to_string(),
            code: r#"fn linear_search<T: PartialEq>(arr: &[T], target: &T) -> Option<usize> {
    for (index, element) in arr.iter().enumerate() {
        if element == target {
            return Some(index);
        }
    }
    None
}

// Generic version with predicate
fn linear_search_with_predicate<T, F>(arr: &[T], predicate: F) -> Option<usize>
where
    F: Fn(&T) -> bool,
{
    for (index, element) in arr.iter().enumerate() {
        if predicate(element) {
            return Some(index);
        }
    }
    None
}"#.to_string(),
            description: "Sequential search through array elements with generic implementation".to_string(),
            algorithm_type: AlgorithmType::Searching,
            difficulty: Difficulty::Beginner,
            expected_explanation: CodeExplanationExpected {
                overview: "Linear search examines each element sequentially until the target is found or the array is exhausted.".to_string(),
                step_by_step: vec![
                    "Iterate through array elements using enumerate for index tracking".to_string(),
                    "Compare each element with the target using PartialEq trait".to_string(),
                    "Return Some(index) immediately when target is found".to_string(),
                    "Return None if loop completes without finding target".to_string(),
                    "Generic version allows custom comparison predicates".to_string(),
                ],
                complexity_analysis: "Time: O(n) - must potentially check every element, Space: O(1) - constant space usage".to_string(),
                key_insights: vec![
                    "Works on unsorted arrays unlike binary search".to_string(),
                    "Generic implementation works with any comparable type".to_string(),
                    "Predicate version enables flexible search criteria".to_string(),
                ],
                best_practices: vec![
                    "Use generic types for reusability across different data types".to_string(),
                    "Return Option type for safe handling of not-found cases".to_string(),
                    "Consider binary search for sorted data for better performance".to_string(),
                ],
            },
            key_concepts: vec!["linear search".to_string(), "generics".to_string(), "iteration".to_string()],
        });

        // Simple Stack Implementation
        self.add_example(CodeExample {
            name: "Stack Data Structure".to_string(),
            language: "rust".to_string(),
            code: r#"pub struct Stack<T> {
    items: Vec<T>,
}

impl<T> Stack<T> {
    pub fn new() -> Self {
        Stack { items: Vec::new() }
    }
    
    pub fn push(&mut self, item: T) {
        self.items.push(item);
    }
    
    pub fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }
    
    pub fn peek(&self) -> Option<&T> {
        self.items.last()
    }
    
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    
    pub fn len(&self) -> usize {
        self.items.len()
    }
}"#.to_string(),
            description: "Generic stack data structure with LIFO (Last In, First Out) operations".to_string(),
            algorithm_type: AlgorithmType::DataStructure,
            difficulty: Difficulty::Beginner,
            expected_explanation: CodeExplanationExpected {
                overview: "Stack implements LIFO data structure using Vec as underlying storage, providing push, pop, and peek operations.".to_string(),
                step_by_step: vec![
                    "Define generic struct wrapping Vec for type flexibility".to_string(),
                    "Implement constructor creating empty stack".to_string(),
                    "Push operation adds element to top (end of Vec)".to_string(),
                    "Pop operation removes and returns top element".to_string(),
                    "Peek allows viewing top element without removal".to_string(),
                    "Utility methods check emptiness and size".to_string(),
                ],
                complexity_analysis: "All operations: O(1) amortized - constant time for push/pop with occasional Vec reallocation".to_string(),
                key_insights: vec![
                    "Vec provides efficient LIFO operations at the end".to_string(),
                    "Generic implementation works with any type".to_string(),
                    "Option types handle empty stack cases safely".to_string(),
                ],
                best_practices: vec![
                    "Use generic types for maximum reusability".to_string(),
                    "Return Option for operations that might fail".to_string(),
                    "Provide both mutable and immutable access methods".to_string(),
                ],
            },
            key_concepts: vec!["data structure".to_string(), "LIFO".to_string(), "generics".to_string()],
        });

        // Two Sum Problem
        self.add_example(CodeExample {
            name: "Two Sum Problem".to_string(),
            language: "rust".to_string(),
            code: r#"use std::collections::HashMap;

fn two_sum(nums: &[i32], target: i32) -> Option<(usize, usize)> {
    let mut map = HashMap::new();
    
    for (i, &num) in nums.iter().enumerate() {
        let complement = target - num;
        
        if let Some(&j) = map.get(&complement) {
            return Some((j, i));
        }
        
        map.insert(num, i);
    }
    
    None
}

// Brute force version for comparison
fn two_sum_brute_force(nums: &[i32], target: i32) -> Option<(usize, usize)> {
    for i in 0..nums.len() {
        for j in (i + 1)..nums.len() {
            if nums[i] + nums[j] == target {
                return Some((i, j));
            }
        }
    }
    None
}"#.to_string(),
            description: "Find two numbers in array that sum to target value using hash map optimization".to_string(),
            algorithm_type: AlgorithmType::Searching,
            difficulty: Difficulty::Intermediate,
            expected_explanation: CodeExplanationExpected {
                overview: "Two Sum problem demonstrates hash map optimization to reduce time complexity from O(n²) to O(n) by trading space for time.".to_string(),
                step_by_step: vec![
                    "Create HashMap to store seen numbers and their indices".to_string(),
                    "For each number, calculate its complement (target - current)".to_string(),
                    "Check if complement exists in the hash map".to_string(),
                    "If found, return the pair of indices".to_string(),
                    "Otherwise, store current number and index in map".to_string(),
                    "Brute force version shows O(n²) nested loop approach".to_string(),
                ],
                complexity_analysis: "Optimized: Time O(n), Space O(n). Brute force: Time O(n²), Space O(1) - hash map trades space for time".to_string(),
                key_insights: vec![
                    "Hash map enables O(1) lookup for complement values".to_string(),
                    "Single pass through array sufficient with proper data structure".to_string(),
                    "Space-time tradeoff: use more memory for faster execution".to_string(),
                ],
                best_practices: vec![
                    "Use HashMap for fast lookups in array problems".to_string(),
                    "Consider complement/difference patterns in sum problems".to_string(),
                    "Compare brute force vs optimized solutions for learning".to_string(),
                ],
            },
            key_concepts: vec!["hash map".to_string(), "two pointers".to_string(), "optimization".to_string()],
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_creation() {
        let dataset = CodeExamplesDataset::new();
        assert!(!dataset.is_empty());
        assert!(dataset.len() > 0);
    }

    #[test]
    fn test_filter_by_algorithm_type() {
        let dataset = CodeExamplesDataset::new();
        let sorting_examples = dataset.get_examples_by_type(AlgorithmType::Sorting);
        assert!(!sorting_examples.is_empty());
        
        for example in sorting_examples {
            assert_eq!(example.algorithm_type, AlgorithmType::Sorting);
        }
    }

    #[test]
    fn test_filter_by_difficulty() {
        let dataset = CodeExamplesDataset::new();
        let beginner_examples = dataset.get_examples_by_difficulty(Difficulty::Beginner);
        assert!(!beginner_examples.is_empty());
        
        for example in beginner_examples {
            assert_eq!(example.difficulty, Difficulty::Beginner);
        }
    }

    #[test]
    fn test_filter_by_language() {
        let dataset = CodeExamplesDataset::new();
        let rust_examples = dataset.get_examples_by_language("rust");
        assert!(!rust_examples.is_empty());
        
        for example in rust_examples {
            assert_eq!(example.language, "rust");
        }
    }

    #[test]
    fn test_add_example() {
        let mut dataset = CodeExamplesDataset::new();
        let initial_len = dataset.len();
        
        let new_example = CodeExample {
            name: "Test Example".to_string(),
            language: "rust".to_string(),
            code: "fn test() {}".to_string(),
            description: "Test description".to_string(),
            algorithm_type: AlgorithmType::Mathematical,
            difficulty: Difficulty::Beginner,
            expected_explanation: CodeExplanationExpected {
                overview: "Test overview".to_string(),
                step_by_step: vec!["Step 1".to_string()],
                complexity_analysis: "O(1)".to_string(),
                key_insights: vec!["Insight 1".to_string()],
                best_practices: vec!["Practice 1".to_string()],
            },
            key_concepts: vec!["test".to_string()],
        };
        
        dataset.add_example(new_example);
        assert_eq!(dataset.len(), initial_len + 1);
    }

    #[test]
    fn test_example_content() {
        let dataset = CodeExamplesDataset::new();
        let examples = dataset.get_examples();
        
        // Check that we have the expected examples
        let binary_search = examples.iter().find(|e| e.name == "Binary Search");
        assert!(binary_search.is_some());
        
        let bubble_sort = examples.iter().find(|e| e.name == "Bubble Sort");
        assert!(bubble_sort.is_some());
        
        let fibonacci = examples.iter().find(|e| e.name == "Fibonacci Recursive");
        assert!(fibonacci.is_some());
    }
}