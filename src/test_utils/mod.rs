//! Test utilities for signal-kit
//!
//! This module provides common helper functions for tests across the codebase.

use std::env;

/// Helper function to check if plotting is enabled via PLOT environment variable
///
/// Returns `true` if the `PLOT` environment variable is set to "true" (case-insensitive),
/// otherwise returns `false`.
///
/// # Example
///
/// ```ignore
/// use crate::test_utils::should_plot;
///
/// #[test]
/// fn my_test() {
///     if !should_plot() {
///         println!("Skipping plot (set PLOT=true to enable)");
///         return;
///     }
///     // ... plotting code ...
/// }
/// ```
pub fn should_plot() -> bool {
    env::var("PLOT")
        .unwrap_or_else(|_| "false".to_string())
        .to_lowercase()
        == "true"
}