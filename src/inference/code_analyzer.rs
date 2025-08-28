//! # Code Analysis System
//!
//! Provides code parsing, analysis, and step-by-step explanation generation
//! with reasoning about algorithm complexity and code patterns.

use crate::utils::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents different types of code constructs
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CodeConstruct {
    Function,
    Loop,
    Conditional,
    Variable,
    DataStructure,
    Algorithm,
    Class,
    Method,
}

/// Complexity analysis for algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Complexity {
    Constant,      // O(1)
    Logarithmic,   // O(log n)
    Linear,        // O(n)
    Linearithmic,  // O(n log n)
    Quadratic,     // O(n²)
    Cubic,         // O(n³)
    Exponential,   // O(2^n)
    Unknown,
}

impl std::fmt::Display for Complexity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Complexity::Constant => write!(f, "O(1)"),
            Complexity::Logarithmic => write!(f, "O(log n)"),
            Complexity::Linear => write!(f, "O(n)"),
            Complexity::Linearithmic => write!(f, "O(n log n)"),
            Complexity::Quadratic => write!(f, "O(n²)"),
            Complexity::Cubic => write!(f, "O(n³)"),
            Complexity::Exponential => write!(f, "O(2^n)"),
            Complexity::Unknown => write!(f, "O(?)"),
        }
    }
}

/// Code pattern identification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CodePattern {
    Iterator,
    Recursion,
    DivideAndConquer,
    DynamicProgramming,
    Greedy,
    TwoPointers,
    SlidingWindow,
    BinarySearch,
    Sorting,
    Hashing,
    Unknown,
}

/// Best practices and code quality indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeQuality {
    pub readability_score: f32,      // 0.0 to 1.0
    pub maintainability_score: f32,  // 0.0 to 1.0
    pub efficiency_score: f32,       // 0.0 to 1.0
    pub best_practices: Vec<String>,
    pub potential_issues: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Represents a parsed code element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeElement {
    pub construct_type: CodeConstruct,
    pub name: String,
    pub line_start: usize,
    pub line_end: usize,
    pub description: String,
    pub purpose: String,
}

/// Analysis result for a piece of code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeAnalysis {
    pub language: String,
    pub elements: Vec<CodeElement>,
    pub time_complexity: Complexity,
    pub space_complexity: Complexity,
    pub patterns: Vec<CodePattern>,
    pub quality: CodeQuality,
    pub explanation_steps: Vec<String>,
}

/// Step-by-step code explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExplanation {
    pub overview: String,
    pub steps: Vec<ExplanationStep>,
    pub complexity_analysis: String,
    pub pattern_analysis: String,
    pub best_practices_notes: Vec<String>,
}

/// Individual step in code explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationStep {
    pub step_number: usize,
    pub code_section: String,
    pub explanation: String,
    pub reasoning: String,
    pub key_concepts: Vec<String>,
}

/// Main code analyzer that provides parsing and analysis capabilities
pub struct CodeAnalyzer {
    /// Known patterns for different programming languages
    language_patterns: HashMap<String, Vec<String>>,
    /// Common algorithm patterns
    algorithm_patterns: HashMap<String, CodePattern>,
}

impl Default for CodeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeAnalyzer {
    /// Create a new code analyzer with predefined patterns
    pub fn new() -> Self {
        let mut language_patterns = HashMap::new();
        let mut algorithm_patterns = HashMap::new();

        // Add common language patterns
        language_patterns.insert(
            "rust".to_string(),
            vec![
                "fn ".to_string(),
                "let ".to_string(),
                "for ".to_string(),
                "while ".to_string(),
                "if ".to_string(),
                "match ".to_string(),
                "struct ".to_string(),
                "impl ".to_string(),
            ],
        );

        language_patterns.insert(
            "python".to_string(),
            vec![
                "def ".to_string(),
                "class ".to_string(),
                "for ".to_string(),
                "while ".to_string(),
                "if ".to_string(),
                "elif ".to_string(),
                "else:".to_string(),
            ],
        );

        // Add algorithm patterns
        algorithm_patterns.insert("binary_search".to_string(), CodePattern::BinarySearch);
        algorithm_patterns.insert("merge_sort".to_string(), CodePattern::DivideAndConquer);
        algorithm_patterns.insert("quick_sort".to_string(), CodePattern::DivideAndConquer);
        algorithm_patterns.insert("fibonacci".to_string(), CodePattern::DynamicProgramming);
        algorithm_patterns.insert("two_sum".to_string(), CodePattern::TwoPointers);

        Self {
            language_patterns,
            algorithm_patterns,
        }
    }

    /// Analyze a piece of code and return comprehensive analysis
    pub fn analyze_code(&self, code: &str, language: &str) -> Result<CodeAnalysis> {
        let elements = self.parse_code_elements(code, language)?;
        let time_complexity = self.analyze_time_complexity(code);
        let space_complexity = self.analyze_space_complexity(code);
        let patterns = self.identify_patterns(code);
        let quality = self.assess_code_quality(code);
        let explanation_steps = self.generate_explanation_steps(code, &elements);

        Ok(CodeAnalysis {
            language: language.to_string(),
            elements,
            time_complexity,
            space_complexity,
            patterns,
            quality,
            explanation_steps,
        })
    }

    /// Generate step-by-step explanation of code
    pub fn explain_code(&self, code: &str, language: &str) -> Result<CodeExplanation> {
        let analysis = self.analyze_code(code, language)?;
        
        let overview = self.generate_overview(code, &analysis);
        let steps = self.create_explanation_steps(code, &analysis);
        let complexity_analysis = format!(
            "Time Complexity: {}, Space Complexity: {}",
            analysis.time_complexity, analysis.space_complexity
        );
        let pattern_analysis = self.explain_patterns(&analysis.patterns);
        let best_practices_notes = analysis.quality.best_practices.clone();

        Ok(CodeExplanation {
            overview,
            steps,
            complexity_analysis,
            pattern_analysis,
            best_practices_notes,
        })
    }

    /// Parse code into structural elements
    fn parse_code_elements(&self, code: &str, language: &str) -> Result<Vec<CodeElement>> {
        let mut elements = Vec::new();
        let lines: Vec<&str> = code.lines().collect();

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            
            // Detect functions
            if self.is_function_declaration(trimmed, language) {
                elements.push(CodeElement {
                    construct_type: CodeConstruct::Function,
                    name: self.extract_function_name(trimmed, language),
                    line_start: line_num + 1,
                    line_end: line_num + 1, // Simplified - would need proper parsing
                    description: format!("Function declaration: {}", trimmed),
                    purpose: "Defines a reusable block of code".to_string(),
                });
            }

            // Detect loops
            if self.is_loop(trimmed, language) {
                elements.push(CodeElement {
                    construct_type: CodeConstruct::Loop,
                    name: "loop".to_string(),
                    line_start: line_num + 1,
                    line_end: line_num + 1,
                    description: format!("Loop construct: {}", trimmed),
                    purpose: "Iterates over a collection or repeats code".to_string(),
                });
            }

            // Detect conditionals
            if self.is_conditional(trimmed, language) {
                elements.push(CodeElement {
                    construct_type: CodeConstruct::Conditional,
                    name: "conditional".to_string(),
                    line_start: line_num + 1,
                    line_end: line_num + 1,
                    description: format!("Conditional statement: {}", trimmed),
                    purpose: "Controls program flow based on conditions".to_string(),
                });
            }
        }

        Ok(elements)
    }

    /// Analyze time complexity of the code
    fn analyze_time_complexity(&self, code: &str) -> Complexity {
        let code_lower = code.to_lowercase();
        
        // Simple heuristic-based complexity analysis
        if code_lower.contains("binary_search") || code_lower.contains("log") {
            Complexity::Logarithmic
        } else if self.count_nested_loops(code) >= 2 {
            Complexity::Quadratic
        } else if self.count_nested_loops(code) == 1 {
            Complexity::Linear
        } else if code_lower.contains("recursive") && code_lower.contains("fibonacci") {
            Complexity::Exponential
        } else if code_lower.contains("sort") && code_lower.contains("merge") {
            Complexity::Linearithmic
        } else {
            Complexity::Linear // Default assumption
        }
    }

    /// Analyze space complexity of the code
    fn analyze_space_complexity(&self, code: &str) -> Complexity {
        let code_lower = code.to_lowercase();
        
        if code_lower.contains("recursive") {
            Complexity::Linear // Stack space for recursion
        } else if code_lower.contains("vec!") || code_lower.contains("array") {
            Complexity::Linear // Additional data structures
        } else {
            Complexity::Constant // In-place operations
        }
    }

    /// Identify algorithmic patterns in the code
    fn identify_patterns(&self, code: &str) -> Vec<CodePattern> {
        let mut patterns = Vec::new();
        let code_lower = code.to_lowercase();

        if code_lower.contains("for") || code_lower.contains("while") {
            patterns.push(CodePattern::Iterator);
        }

        if code_lower.contains("recursive") || code_lower.contains("fn") && code_lower.contains("self") {
            patterns.push(CodePattern::Recursion);
        }

        if code_lower.contains("binary_search") || code_lower.contains("mid") {
            patterns.push(CodePattern::BinarySearch);
        }

        if code_lower.contains("sort") {
            patterns.push(CodePattern::Sorting);
        }

        if code_lower.contains("hashmap") || code_lower.contains("hash") {
            patterns.push(CodePattern::Hashing);
        }

        if patterns.is_empty() {
            patterns.push(CodePattern::Unknown);
        }

        patterns
    }

    /// Assess code quality and best practices
    fn assess_code_quality(&self, code: &str) -> CodeQuality {
        let mut best_practices = Vec::new();
        let mut potential_issues = Vec::new();
        let mut suggestions = Vec::new();

        // Check for good practices
        if code.contains("//") || code.contains("///") {
            best_practices.push("Code includes comments for documentation".to_string());
        } else {
            potential_issues.push("Code lacks comments for clarity".to_string());
            suggestions.push("Add comments to explain complex logic".to_string());
        }

        if code.contains("fn ") && code.lines().count() < 20 {
            best_practices.push("Functions are reasonably sized".to_string());
        }

        if code.contains("Result<") || code.contains("Option<") {
            best_practices.push("Uses Rust's error handling patterns".to_string());
        }

        // Simple scoring based on heuristics
        let readability_score = if code.contains("//") { 0.8 } else { 0.4 };
        let maintainability_score = if code.lines().count() < 50 { 0.7 } else { 0.5 };
        let efficiency_score = if self.count_nested_loops(code) <= 1 { 0.8 } else { 0.6 };

        CodeQuality {
            readability_score,
            maintainability_score,
            efficiency_score,
            best_practices,
            potential_issues,
            suggestions,
        }
    }

    /// Generate high-level explanation steps
    fn generate_explanation_steps(&self, _code: &str, elements: &[CodeElement]) -> Vec<String> {
        let mut steps = Vec::new();
        
        steps.push("1. Parse the code structure and identify main components".to_string());
        
        if elements.iter().any(|e| e.construct_type == CodeConstruct::Function) {
            steps.push("2. Analyze function definitions and their purposes".to_string());
        }
        
        if elements.iter().any(|e| e.construct_type == CodeConstruct::Loop) {
            steps.push("3. Examine loop structures and iteration patterns".to_string());
        }
        
        if elements.iter().any(|e| e.construct_type == CodeConstruct::Conditional) {
            steps.push("4. Review conditional logic and decision points".to_string());
        }
        
        steps.push("5. Analyze algorithm complexity and efficiency".to_string());
        steps.push("6. Identify patterns and best practices used".to_string());

        steps
    }

    /// Helper methods for code analysis
    fn is_function_declaration(&self, line: &str, language: &str) -> bool {
        match language {
            "rust" => line.starts_with("fn ") || line.contains("fn "),
            "python" => line.starts_with("def "),
            _ => line.contains("function") || line.contains("def"),
        }
    }

    fn extract_function_name(&self, line: &str, language: &str) -> String {
        match language {
            "rust" => {
                if let Some(start) = line.find("fn ") {
                    let after_fn = &line[start + 3..];
                    if let Some(end) = after_fn.find('(') {
                        after_fn[..end].trim().to_string()
                    } else {
                        "unknown".to_string()
                    }
                } else {
                    "unknown".to_string()
                }
            }
            "python" => {
                if let Some(start) = line.find("def ") {
                    let after_def = &line[start + 4..];
                    if let Some(end) = after_def.find('(') {
                        after_def[..end].trim().to_string()
                    } else {
                        "unknown".to_string()
                    }
                } else {
                    "unknown".to_string()
                }
            }
            _ => "unknown".to_string(),
        }
    }

    fn is_loop(&self, line: &str, _language: &str) -> bool {
        line.contains("for ") || line.contains("while ") || line.contains("loop")
    }

    fn is_conditional(&self, line: &str, _language: &str) -> bool {
        line.starts_with("if ") || line.contains("if ") || line.contains("match ")
    }

    fn count_nested_loops(&self, code: &str) -> usize {
        let mut max_nesting = 0;
        let mut current_nesting = 0;

        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.contains("for ") || trimmed.contains("while ") {
                current_nesting += 1;
                max_nesting = max_nesting.max(current_nesting);
            }
            if trimmed == "}" && current_nesting > 0 {
                current_nesting -= 1;
            }
        }

        max_nesting
    }

    fn generate_overview(&self, _code: &str, analysis: &CodeAnalysis) -> String {
        format!(
            "This {} code implements an algorithm with {} time complexity and {} space complexity. \
            It contains {} code elements and demonstrates {} patterns.",
            analysis.language,
            analysis.time_complexity,
            analysis.space_complexity,
            analysis.elements.len(),
            analysis.patterns.len()
        )
    }

    fn create_explanation_steps(&self, code: &str, analysis: &CodeAnalysis) -> Vec<ExplanationStep> {
        let mut steps = Vec::new();
        let lines: Vec<&str> = code.lines().collect();

        // Create steps based on code elements
        for (i, element) in analysis.elements.iter().enumerate() {
            let code_section = if element.line_start <= lines.len() {
                lines[element.line_start - 1].to_string()
            } else {
                "Code section".to_string()
            };

            steps.push(ExplanationStep {
                step_number: i + 1,
                code_section,
                explanation: element.description.clone(),
                reasoning: element.purpose.clone(),
                key_concepts: vec![format!("{:?}", element.construct_type)],
            });
        }

        steps
    }

    fn explain_patterns(&self, patterns: &[CodePattern]) -> String {
        if patterns.is_empty() {
            return "No specific algorithmic patterns identified.".to_string();
        }

        let pattern_descriptions: Vec<String> = patterns
            .iter()
            .map(|p| match p {
                CodePattern::Iterator => "Uses iteration to process elements sequentially".to_string(),
                CodePattern::Recursion => "Employs recursive approach for problem decomposition".to_string(),
                CodePattern::BinarySearch => "Implements binary search for efficient searching".to_string(),
                CodePattern::Sorting => "Contains sorting algorithm implementation".to_string(),
                CodePattern::Hashing => "Uses hash-based data structures for fast lookups".to_string(),
                _ => format!("Pattern: {:?}", p),
            })
            .collect();

        format!("Identified patterns: {}", pattern_descriptions.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_analyzer_creation() {
        let analyzer = CodeAnalyzer::new();
        assert!(!analyzer.language_patterns.is_empty());
        assert!(!analyzer.algorithm_patterns.is_empty());
    }

    #[test]
    fn test_complexity_display() {
        assert_eq!(Complexity::Linear.to_string(), "O(n)");
        assert_eq!(Complexity::Quadratic.to_string(), "O(n²)");
        assert_eq!(Complexity::Constant.to_string(), "O(1)");
    }

    #[test]
    fn test_function_detection() {
        let analyzer = CodeAnalyzer::new();
        assert!(analyzer.is_function_declaration("fn main() {", "rust"));
        assert!(analyzer.is_function_declaration("def hello():", "python"));
        assert!(!analyzer.is_function_declaration("let x = 5;", "rust"));
    }

    #[test]
    fn test_function_name_extraction() {
        let analyzer = CodeAnalyzer::new();
        assert_eq!(analyzer.extract_function_name("fn main() {", "rust"), "main");
        assert_eq!(analyzer.extract_function_name("def hello():", "python"), "hello");
    }

    #[test]
    fn test_loop_detection() {
        let analyzer = CodeAnalyzer::new();
        assert!(analyzer.is_loop("for i in 0..10 {", "rust"));
        assert!(analyzer.is_loop("while x > 0 {", "rust"));
        assert!(!analyzer.is_loop("let x = 5;", "rust"));
    }

    #[test]
    fn test_nested_loop_counting() {
        let analyzer = CodeAnalyzer::new();
        let code = r#"
            for i in 0..10 {
                for j in 0..10 {
                    println!("{}", i + j);
                }
            }
        "#;
        assert_eq!(analyzer.count_nested_loops(code), 2);
    }

    #[test]
    fn test_pattern_identification() {
        let analyzer = CodeAnalyzer::new();
        let code = "for i in 0..10 { binary_search(arr, target); }";
        let patterns = analyzer.identify_patterns(code);
        assert!(patterns.contains(&CodePattern::Iterator));
        assert!(patterns.contains(&CodePattern::BinarySearch));
    }
}