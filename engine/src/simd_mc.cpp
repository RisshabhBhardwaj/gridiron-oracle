// engine/src/simd_mc.cpp
//
// SIMD Monte Carlo Sampler — Implementation.
// See engine/include/simd_mc.hpp for API documentation.

#include "simd_mc.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace gridiron {

// ── Constructor
// ───────────────────────────────────────────────────────────────

SIMDMonteCarlo::SIMDMonteCarlo(uint32_t seed)
    : rng_(seed), uniform_dist_(0.0f, 1.0f) {}

// ── Box-Muller Pair
// ────────────────────────────────────────────────────────────

void SIMDMonteCarlo::box_muller_pair(float u1, float u2, float *z1, float *z2) {
  // Standard Box-Muller transform.
  // z1, z2 ∼ N(0, 1).
  float r = std::sqrt(-2.0f * std::log(std::max(u1, 1e-38f)));
  float theta = 2.0f * static_cast<float>(M_PI) * u2;
  *z1 = r * std::cos(theta);
  *z2 = r * std::sin(theta);
}

// ── batch_uniform
// ──────────────────────────────────────────────────────────────

void SIMDMonteCarlo::batch_uniform(float *out, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) {
    out[i] = uniform_dist_(rng_);
  }
}

// ── batch_normal
// ──────────────────────────────────────────────────────────────

#ifdef GRIDIRON_NEON

void SIMDMonteCarlo::neon_box_muller(float *out, uint32_t n) {
  // Process 4 pairs (8 samples) at a time using NEON.
  uint32_t i = 0;
  const uint32_t n_blocks = (n / 8) * 8;

  for (; i < n_blocks; i += 8) {
    // Draw 8 uniforms
    float u[8];
    for (float &v : u)
      v = uniform_dist_(rng_);

    // log(u1) using scalar (no NEON log)
    float log_u1[4], theta[4], z1[4], z2[4];
    for (int k = 0; k < 4; ++k) {
      log_u1[k] = std::log(std::max(u[k], 1e-38f));
      theta[k] = 2.0f * M_PI * u[k + 4];
      float r = std::sqrt(-2.0f * log_u1[k]);
      z1[k] = r * std::cos(theta[k]);
      z2[k] = r * std::sin(theta[k]);
    }
    vst1q_f32(out + i, vld1q_f32(z1));
    vst1q_f32(out + i + 4, vld1q_f32(z2));
  }

  // Scalar tail
  while (i < n) {
    float u1 = uniform_dist_(rng_), u2 = uniform_dist_(rng_);
    float z1, z2;
    box_muller_pair(u1, u2, &z1, &z2);
    out[i++] = z1;
    if (i < n)
      out[i++] = z2;
  }
}

#endif // GRIDIRON_NEON

void SIMDMonteCarlo::batch_normal(float *out, uint32_t n, float mean,
                                  float std) {
  // Draw n samples from N(mean, std²).
#ifdef GRIDIRON_NEON
  neon_box_muller(out, n);
  // Scale: out[i] = mean + std * z[i]
  const uint32_t n4 = (n / 4) * 4;
  float32x4_t mean4 = vdupq_n_f32(mean);
  float32x4_t std4 = vdupq_n_f32(std);
  for (uint32_t i = 0; i < n4; i += 4) {
    float32x4_t z = vld1q_f32(out + i);
    vst1q_f32(out + i, vaddq_f32(mean4, vmulq_f32(std4, z)));
  }
  for (uint32_t i = n4; i < n; ++i) {
    out[i] = mean + std * out[i];
  }
#else
  uint32_t i = 0;
  for (; i + 1 < n; i += 2) {
    float u1 = uniform_dist_(rng_), u2 = uniform_dist_(rng_);
    float z1, z2;
    box_muller_pair(u1, u2, &z1, &z2);
    out[i] = mean + std * z1;
    out[i + 1] = mean + std * z2;
  }
  if (i < n) {
    float u1 = uniform_dist_(rng_), u2 = uniform_dist_(rng_);
    float z1, z2;
    box_muller_pair(u1, u2, &z1, &z2);
    out[i] = mean + std * z1;
  }
#endif
}

// ── batch_normal_clipped
// ──────────────────────────────────────────────────────

void SIMDMonteCarlo::batch_normal_clipped(float *out, uint32_t n, float mean,
                                          float std, float clip_min) {
  batch_normal(out, n, mean, std);
  for (uint32_t i = 0; i < n; ++i) {
    if (out[i] < clip_min)
      out[i] = clip_min;
  }
}

// ── Gamma Sample (Marsaglia-Tsang) ──────────────────────────────────────────

float SIMDMonteCarlo::gamma_sample(float alpha) {
  // Marsaglia-Tsang method (2000) for Gamma(alpha, 1).
  // Valid for alpha >= 1. For alpha < 1: sample Gamma(alpha+1) then
  // multiply by U^(1/alpha).
  if (alpha < 1.0f) {
    float g = gamma_sample(alpha + 1.0f);
    float u = uniform_dist_(rng_);
    return g * std::pow(u, 1.0f / alpha);
  }

  float d = alpha - 1.0f / 3.0f;
  float c = 1.0f / std::sqrt(9.0f * d);

  // Expected iterations ≈ 1.3 for typical alpha values (Marsaglia & Tsang 2000).
  // Cap at 1000 to bound worst-case latency; fall back to the mode d = alpha-1/3,
  // which is the MAP estimate and a safe conservative proxy for the mean.
  constexpr int kMaxIter = 1000;
  for (int iter = 0; iter < kMaxIter; ++iter) {
    float x, v;
    do {
      float u1 = uniform_dist_(rng_), u2 = uniform_dist_(rng_);
      float z, dummy;
      box_muller_pair(u1, u2, &z, &dummy);
      v = 1.0f + c * z;
      x = z;
    } while (v <= 0.0f);

    v = v * v * v;
    float u = uniform_dist_(rng_);

    // Accept-reject test
    if (u < 1.0f - 0.0331f * (x * x) * (x * x)) {
      return d * v;
    }
    if (std::log(u) < 0.5f * x * x + d * (1.0f - v + std::log(v))) {
      return d * v;
    }
  }
  // Deterministic fallback: mode of Gamma(alpha,1) = alpha - 1 (mapped via d).
  return d;
}

// ── batch_dirichlet
// ───────────────────────────────────────────────────────────

void SIMDMonteCarlo::batch_dirichlet(float *out, uint32_t n_draws,
                                     const float *alphas, uint32_t n_cats) {
  // Dirichlet(alpha) = normalize(Gamma(alpha_k) for k in 1..n_cats).
  for (uint32_t draw = 0; draw < n_draws; ++draw) {
    float sum = 0.0f;
    float *row = out + draw * n_cats;
    for (uint32_t k = 0; k < n_cats; ++k) {
      row[k] = gamma_sample(std::max(alphas[k], 1e-4f));
      sum += row[k];
    }
    if (sum < 1e-10f)
      sum = 1.0f; // avoid div-by-zero
    for (uint32_t k = 0; k < n_cats; ++k) {
      row[k] /= sum;
    }
  }
}

// ── batch_multivariate_normal
// ─────────────────────────────────────────────────

void SIMDMonteCarlo::batch_multivariate_normal(float *out, uint32_t n_draws,
                                               const float *means,
                                               const float *chol_L,
                                               uint32_t n_vars) {
  // out[draw × n_vars] = means + L × z  where z ∼ N(0, I)
  std::vector<float> z(n_vars);
  for (uint32_t d = 0; d < n_draws; ++d) {
    // Draw z ∼ N(0, I)
    batch_normal(z.data(), n_vars, 0.0f, 1.0f);
    // y = L × z (lower-triangular multiply)
    float *row = out + d * n_vars;
    for (uint32_t i = 0; i < n_vars; ++i) {
      float acc = 0.0f;
      for (uint32_t k = 0; k <= i; ++k) {
        acc += chol_L[i * n_vars + k] * z[k];
      }
      row[i] = means[i] + acc;
    }
  }
}

} // namespace gridiron

// ── C API
// ──────────────────────────────────────────────────────────────────────

extern "C" {

void *simd_mc_create(uint32_t seed) {
  return new gridiron::SIMDMonteCarlo(seed);
}

void simd_mc_destroy(void *handle) {
  delete static_cast<gridiron::SIMDMonteCarlo *>(handle);
}

void simd_normal_sample(void *handle, float *out, uint32_t n, float mean,
                        float std) {
  static_cast<gridiron::SIMDMonteCarlo *>(handle)->batch_normal(out, n, mean,
                                                                std);
}

void simd_normal_clipped(void *handle, float *out, uint32_t n, float mean,
                         float std, float clip_min) {
  static_cast<gridiron::SIMDMonteCarlo *>(handle)->batch_normal_clipped(
      out, n, mean, std, clip_min);
}

void simd_dirichlet_sample(void *handle, float *out, uint32_t n_draws,
                           const float *alphas, uint32_t n_cats) {
  static_cast<gridiron::SIMDMonteCarlo *>(handle)->batch_dirichlet(
      out, n_draws, alphas, n_cats);
}

} // extern "C"
