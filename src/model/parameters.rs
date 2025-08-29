/*!
Parameter Registry and Introspection Utilities

This module provides a formal `Parameter` trait and lightweight collection
types that make it easy to expose, traverse, and mutate model parameters
(e.g., embeddings, linear layers, layer norms) without a full autograd stack.

Goals
- Stable parameter naming for optimizer state and checkpointing
- Borrow-safe mutable traversal over parameter buffers
- Utilities to collect parameters from common containers (`Vec<f32>`, `Vec<Vec<f32>>`)
- Type-safe helpers (ParameterRef/ParameterMut) instead of requiring trait objects

Typical usage in a model:
- Expose parameters via `parameters_mut()` (returns `Vec<ParameterMut>`)
- Or provide a convenience iterator `for_each_parameter(|name, slice| { ... })`
- Use stable names like:
  - "embeddings.weight[`<row>`]"
  - "lm_head.weight[`<row>`]"
  - "lm_head.bias"
*/

use core::fmt;

/// A formal trait for a trainable parameter buffer.
///
/// Notes:
/// - `as_slice` provides immutable access to the contiguous `f32` buffer.
/// - `as_mut_slice` provides mutable access so optimizers can update in place.
///
/// The trait is intentionally minimal and object-safe, enabling
/// heterogeneous parameter traversal if needed.
pub trait Parameter {
    /// A stable, unique name for this parameter buffer.
    fn name(&self) -> &str;

    /// Immutable view of the parameter data.
    fn as_slice(&self) -> &[f32];

    /// Mutable view of the parameter data.
    fn as_mut_slice(&mut self) -> &mut [f32];
}

/// Immutable handle to a named parameter slice.
pub struct ParameterRef<'a> {
    name: String,
    data: &'a [f32],
}

impl<'a> ParameterRef<'a> {
    /// Create a new immutable parameter reference.
    pub fn new<N: Into<String>>(name: N, data: &'a [f32]) -> Self {
        Self {
            name: name.into(),
            data,
        }
    }

    /// Parameter name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Immutable data slice.
    pub fn data(&self) -> &[f32] {
        self.data
    }

    /// Current length of the data slice.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a> fmt::Debug for ParameterRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParameterRef")
            .field("name", &self.name)
            .field("len", &self.len())
            .finish()
    }
}

/// Mutable handle to a named parameter slice.
pub struct ParameterMut<'a> {
    name: String,
    data: &'a mut [f32],
}

impl<'a> ParameterMut<'a> {
    /// Create a new mutable parameter reference.
    pub fn new<N: Into<String>>(name: N, data: &'a mut [f32]) -> Self {
        Self {
            name: name.into(),
            data,
        }
    }

    /// Parameter name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Immutable data slice.
    pub fn data(&self) -> &[f32] {
        self.data
    }

    /// Mutable data slice.
    pub fn data_mut(&mut self) -> &mut [f32] {
        self.data
    }

    /// Current length of the data slice.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a> fmt::Debug for ParameterMut<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParameterMut")
            .field("name", &self.name)
            .field("len", &self.len())
            .finish()
    }
}

impl<'a> Parameter for ParameterMut<'a> {
    fn name(&self) -> &str {
        self.name()
    }

    fn as_slice(&self) -> &[f32] {
        self.data()
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        self.data_mut()
    }
}

/// Minimal shape/metadata for a named parameter.
#[derive(Clone, Debug)]
pub struct ParameterInfo {
    /// Stable parameter name.
    pub name: String,
    /// Number of f32 elements in the buffer.
    pub len: usize,
}

impl ParameterInfo {
    pub fn new<N: Into<String>>(name: N, len: usize) -> Self {
        Self {
            name: name.into(),
            len,
        }
    }
}

/// A collection of mutable parameter references, useful as a registry
/// during a single optimization step.
///
/// The registry is a thin wrapper to make lookups easier while preserving
/// borrow safety. It does not own the underlying buffers.
pub struct ParameterRegistryMut<'a> {
    params: Vec<ParameterMut<'a>>,
}

impl<'a> ParameterRegistryMut<'a> {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    /// Add a parameter to the registry.
    pub fn add(&mut self, p: ParameterMut<'a>) {
        self.params.push(p);
    }

    /// Extend with more parameters.
    pub fn extend<I: IntoIterator<Item = ParameterMut<'a>>>(&mut self, iter: I) {
        self.params.extend(iter);
    }

    /// Iterator over all mutable parameters.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut ParameterMut<'a>> {
        self.params.iter_mut()
    }

    /// Iterator over immutable parameter views.
    pub fn iter(&self) -> impl Iterator<Item = &ParameterMut<'a>> {
        self.params.iter()
    }

    /// Get a mutable slice by its name (exact match).
    pub fn by_name_mut(&mut self, name: &str) -> Option<&mut [f32]> {
        self.params
            .iter_mut()
            .find(|p| p.name() == name)
            .map(|p| p.data_mut())
    }

    /// Get the list of parameter infos (name + length), useful for logging.
    pub fn infos(&self) -> Vec<ParameterInfo> {
        self.params
            .iter()
            .map(|p| ParameterInfo::new(p.name().to_string(), p.len()))
            .collect()
    }

    /// Number of parameters in the registry.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether the registry contains no parameters.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

/// Convenience helpers to collect parameters from common containers.
///
/// Create a single immutable parameter reference from a `Vec<f32>`.
pub fn single_ref<'a, N: Into<String>>(name: N, v: &'a Vec<f32>) -> ParameterRef<'a> {
    ParameterRef::new(name, v.as_slice())
}

/// Create a single mutable parameter reference from a `Vec<f32>`.
pub fn single_mut<'a, N: Into<String>>(name: N, v: &'a mut Vec<f32>) -> ParameterMut<'a> {
    ParameterMut::new(name, v.as_mut_slice())
}

/// Collect immutable row parameters from a matrix-like `Vec<Vec<f32>>`,
/// producing names in the form `<base>[row]`.
pub fn collect_rows_ref<'a, N: AsRef<str>>(base: N, rows: &'a [Vec<f32>]) -> Vec<ParameterRef<'a>> {
    let base = base.as_ref();
    rows.iter()
        .enumerate()
        .map(|(i, row)| ParameterRef::new(format!("{}[{}]", base, i), row.as_slice()))
        .collect()
}

/// Collect mutable row parameters from a matrix-like `Vec<Vec<f32>>`,
/// producing names in the form `<base>[row]`.
pub fn collect_rows_mut<'a, N: AsRef<str>>(
    base: N,
    rows: &'a mut [Vec<f32>],
) -> Vec<ParameterMut<'a>> {
    let base = base.as_ref();
    rows.iter_mut()
        .enumerate()
        .map(|(i, row)| ParameterMut::new(format!("{}[{}]", base, i), row.as_mut_slice()))
        .collect()
}

/// Build a `ParameterRegistryMut` from multiple parameter groups conveniently.
///
/// Example:
/// let mut reg = registry_from_groups(vec![
///     collect_rows_mut("lm_head.weight", &mut lm_w),
///     vec![single_mut("lm_head.bias", &mut lm_b)],
///     collect_rows_mut("embeddings.weight", &mut emb_w),
/// ]);
pub fn registry_from_groups<'a>(groups: Vec<Vec<ParameterMut<'a>>>) -> ParameterRegistryMut<'a> {
    let mut reg = ParameterRegistryMut::new();
    for g in groups {
        reg.extend(g);
    }
    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_ref_len() {
        let v = vec![1.0f32, 2.0, 3.0];
        let p = single_ref("layer.bias", &v);
        assert_eq!(p.name(), "layer.bias");
        assert_eq!(p.len(), 3);
        assert_eq!(p.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parameter_mut_edit() {
        let mut v = vec![1.0f32, 2.0, 3.0];
        {
            let mut p = single_mut("layer.bias", &mut v);
            let data = p.data_mut();
            data[0] = 10.0;
            data[2] = -5.0;
        }
        assert_eq!(v, vec![10.0, 2.0, -5.0]);
    }

    #[test]
    fn test_collect_rows_mut_and_registry_lookup() {
        let mut w = vec![vec![1.0f32, 0.0], vec![0.5, -0.3]];
        let mut b = vec![0.1f32, 0.2];

        // Limit the lifetime of the registry so mutable borrows end before assertions
        let infos;
        {
            let mut reg = registry_from_groups(vec![
                collect_rows_mut("linear.weight", &mut w),
                vec![single_mut("linear.bias", &mut b)],
            ]);

            assert_eq!(reg.len(), 3);

            // Update bias via registry
            {
                let bias = reg.by_name_mut("linear.bias").expect("bias not found");
                bias[0] += 1.0;
                bias[1] += 1.0;
            }

            // Update a row via registry
            {
                let row0 = reg.by_name_mut("linear.weight[0]").expect("row0 not found");
                row0[1] = 2.0;
            }

            // Capture infos while registry is alive
            infos = reg.infos();
        } // reg dropped here; mutable borrows released

        assert_eq!(b, vec![1.1, 1.2]);
        assert_eq!(w[0], vec![1.0, 2.0]);

        // Verify infos
        assert!(infos.iter().any(|i| i.name == "linear.bias" && i.len == 2));
        assert!(
            infos
                .iter()
                .any(|i| i.name == "linear.weight[1]" && i.len == 2)
        );
    }

    #[test]
    fn test_collect_rows_ref() {
        let w = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let refs = collect_rows_ref("W", &w);
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].name(), "W[0]");
        assert_eq!(refs[1].name(), "W[1]");
        assert_eq!(refs[0].data(), &[1.0, 2.0, 3.0]);
        assert_eq!(refs[1].data(), &[4.0, 5.0, 6.0]);
    }
}
