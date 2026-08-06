// engine/include/simd_mc.hpp
//
// PHASE 5 STUB — not production code. API and implementation are complete but
// this component is not wired into the main engine pipeline. Python bindings
// (engine/python_bindings.py) call it from ml/monte_carlo.py when available.
// Catch2 coverage lives in engine/tests/simd_mc_test.cpp; keep that coverage in
// place before promoting to production.
//
// SIMD Monte Carlo Sampler — ARM NEON accelerated random sampling.
//
// Vectorizes Box-Muller transform and Dirichlet sampling using ARM NEON
// intrinsics (Apple Silicon M1/M2/M3). Falls back to scalar on x86.
//
// USAGE
// -----
//   #include "simd_mc.hpp"
//   using namespace gridiron;
//
//   // Fill 1024 normal samples with mean=80, std=20
//   std::vector<float> out(1024);
//   SIMDMonteCarlo mc(rng_seed=42);
//   mc.batch_normal(out.data(), 1024, 80.0f, 20.0f);
//
//   // Dirichlet draws for volume redistribution
//   float alphas[3] = {2.0f, 3.0f, 1.0f};
//   mc.batch_dirichlet(out.data(), 1000, alphas, 3);  // 1000 draws, 3 categories
//
// Python ctypes binding:
//   lib.simd_normal_sample(out_ptr, n, mean, std) -> void
//   lib.simd_dirichlet_sample(out_ptr, n_draws, alphas_ptr, n_cats) -> void

#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <random>
#include <vector>

// SIMD detection
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define GRIDIRON_NEON 1
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define GRIDIRON_AVX2 1
#else
#  define GRIDIRON_SCALAR 1
#endif

namespace gridiron {

// ── SIMDMonteCarlo ─────────────────────────────────────────────────────────────

class SIMDMonteCarlo {
public:
    explicit SIMDMonteCarlo(uint32_t seed = 42);
    ~SIMDMonteCarlo() = default;

    // ── Normal sampling ────────────────────────────────────────────────────────

    /// Fill `n` float32 samples from N(mean, std²) into `out`.
    /// Uses vectorized Box-Muller transform.
    /// @param out   Pre-allocated float array of size n
    /// @param n     Number of samples
    /// @param mean  Distribution mean
    /// @param std   Standard deviation (must be > 0)
    void batch_normal(float* out, uint32_t n, float mean, float std);

    /// Fill `n` float32 samples clipped to [clip_min, +∞).
    /// Equivalent to TruncatedNormal(mean, std, a=clip_min).
    /// Primary use: player stat sampling (clips at 0).
    void batch_normal_clipped(
        float*   out,
        uint32_t n,
        float    mean,
        float    std,
        float    clip_min = 0.0f
    );

    // ── Uniform sampling ──────────────────────────────────────────────────────

    /// Fill `n` float32 samples from Uniform(0, 1).
    void batch_uniform(float* out, uint32_t n);

    // ── Dirichlet sampling ────────────────────────────────────────────────────

    /// Draw `n_draws` samples from Dirichlet(alpha) with `n_cats` categories.
    /// Each draw is a `n_cats`-vector that sums to 1.0.
    ///
    /// Output layout: out[draw * n_cats + cat] = probability for category `cat`
    ///   in draw `draw`.
    ///
    /// Algorithm: Dirichlet ≡ normalized Gamma(alpha_k, 1) draws.
    ///   Each Gamma(α, 1) drawn via Marsaglia-Tsang method.
    ///
    /// Primary use: Target share redistribution (volume_redistribution.py).
    ///
    /// @param out      Pre-allocated float array of size n_draws × n_cats
    /// @param n_draws  Number of Dirichlet draws
    /// @param alphas   Concentration parameters [n_cats]
    /// @param n_cats   Number of categories
    void batch_dirichlet(
        float*         out,
        uint32_t       n_draws,
        const float*   alphas,
        uint32_t       n_cats
    );

    // ── Correlated sampling ───────────────────────────────────────────────────

    /// Draw `n_draws` samples from a multivariate normal using pre-computed
    /// Cholesky factor L (lower triangular, row-major).
    ///
    /// @param out     Pre-allocated [n_draws × n_vars] float array
    /// @param n_draws Number of Monte Carlo draws
    /// @param means   Mean vector [n_vars]
    /// @param chol_L  Lower Cholesky factor [n_vars × n_vars], row-major
    /// @param n_vars  Dimension of the distribution
    void batch_multivariate_normal(
        float*         out,
        uint32_t       n_draws,
        const float*   means,
        const float*   chol_L,
        uint32_t       n_vars
    );

private:
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_dist_;

    // Box-Muller: draw 2 standard normal samples from 2 uniforms.
    void box_muller_pair(float u1, float u2, float* z1, float* z2);

    // Gamma(alpha, 1) draw using Marsaglia-Tsang method.
    float gamma_sample(float alpha);

#ifdef GRIDIRON_NEON
    // NEON-vectorized Box-Muller: process 4 pairs at once.
    void neon_box_muller(float* out, uint32_t n);
#endif
};

// ── Convenience C API ──────────────────────────────────────────────────────────

} // namespace gridiron

extern "C" {

/// Create a SIMDMonteCarlo instance.
void* simd_mc_create(uint32_t seed);

/// Destroy a SIMDMonteCarlo instance.
void simd_mc_destroy(void* handle);

/// Fill `n` float32 N(mean, std²) samples. out[n].
void simd_normal_sample(void* handle, float* out, uint32_t n, float mean, float std);

/// Fill `n` truncated-normal samples clipped to [clip_min, ∞). out[n].
void simd_normal_clipped(void* handle, float* out, uint32_t n,
                         float mean, float std, float clip_min);

/// Draw `n_draws` Dirichlet samples with `n_cats` categories.
/// out[n_draws × n_cats].
void simd_dirichlet_sample(void* handle, float* out, uint32_t n_draws,
                            const float* alphas, uint32_t n_cats);

} // extern "C"
