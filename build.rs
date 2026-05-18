fn main() {
    declare_simd_cfgs();
    emit_selected_backend(select_backend());
}

/// Declares the custom cfg names emitted by this build script.
fn declare_simd_cfgs() {
    println!("cargo:rerun-if-env-changed=SIGNAL_KIT_SIMD_BACKEND");
    println!("cargo:rustc-check-cfg=cfg(signal_kit_simd_avx512)");
    println!("cargo:rustc-check-cfg=cfg(signal_kit_simd_avx2)");
    println!("cargo:rustc-check-cfg=cfg(signal_kit_simd_sse2)");
    println!("cargo:rustc-check-cfg=cfg(signal_kit_simd_scalar)");
}

/// Emits the selected backend cfg and an informational environment value.
fn emit_selected_backend(backend: SimdBackend) {
    println!("cargo:rustc-cfg={}", backend.cfg_name());
    println!("cargo:rustc-env=SIGNAL_KIT_SIMD_BACKEND={}", backend.name());
}

/// Selects a backend for the Cargo target using build-host feature detection.
fn select_backend() -> SimdBackend {
    validate_target_arch();
    forced_backend().unwrap_or_else(|| if target_is_x86() { detect_backend() } else { SimdBackend::Scalar })
}

/// Returns a user-forced backend from `SIGNAL_KIT_SIMD_BACKEND`.
fn forced_backend() -> Option<SimdBackend> {
    let value = std::env::var("SIGNAL_KIT_SIMD_BACKEND").ok()?;
    Some(match value.as_str() {
        "avx512" | "avx512f" => SimdBackend::Avx512,
        "avx2" | "avx2_fma" | "avx2-fma" => SimdBackend::Avx2,
        "sse2" => SimdBackend::Sse2,
        "scalar" => SimdBackend::Scalar,
        other => panic!("unsupported SIGNAL_KIT_SIMD_BACKEND value: {other}"),
    })
}

/// Panics when an x86 backend is forced for a non-x86 Cargo target.
fn validate_target_arch() {
    if !target_is_x86() && forced_backend().is_some_and(|backend| backend != SimdBackend::Scalar) {
        panic!("SIMD backends can only be forced for x86 or x86_64 targets");
    }
}

/// Returns true when Cargo is compiling this crate for x86 or x86_64.
fn target_is_x86() -> bool {
    matches!(std::env::var("CARGO_CFG_TARGET_ARCH").as_deref(), Ok("x86" | "x86_64"))
}

/// Detects the highest-priority backend supported by the build host.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_backend() -> SimdBackend {
    if std::is_x86_feature_detected!("avx512f") {
        SimdBackend::Avx512
    } else if has_avx2_fma() {
        SimdBackend::Avx2
    } else if std::is_x86_feature_detected!("sse2") {
        SimdBackend::Sse2
    } else {
        SimdBackend::Scalar
    }
}

/// Returns scalar when the build host cannot run x86 feature detection.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_backend() -> SimdBackend {
    SimdBackend::Scalar
}

/// Returns true when the build host has both AVX2 and FMA.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn has_avx2_fma() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SimdBackend {
    Avx512,
    Avx2,
    Sse2,
    Scalar,
}

impl SimdBackend {
    /// Returns the Rust cfg name for this backend.
    fn cfg_name(self) -> &'static str {
        match self {
            Self::Avx512 => "signal_kit_simd_avx512",
            Self::Avx2 => "signal_kit_simd_avx2",
            Self::Sse2 => "signal_kit_simd_sse2",
            Self::Scalar => "signal_kit_simd_scalar",
        }
    }

    /// Returns a human-readable backend name for build diagnostics.
    fn name(self) -> &'static str {
        match self {
            Self::Avx512 => "avx512f",
            Self::Avx2 => "avx2-fma",
            Self::Sse2 => "sse2",
            Self::Scalar => "scalar",
        }
    }
}
