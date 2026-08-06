/**
 * engine/tests/simd_mc_test.cpp
 *
 * Catch2 v3 unit tests for SIMDMonteCarlo and its C API wrappers.
 *
 * Tests focus on invariants that should hold across scalar / NEON / AVX2
 * implementations rather than architecture-specific sample sequences.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "simd_mc.hpp"

using namespace gridiron;
using Catch::Approx;

TEST_CASE("SIMDMonteCarlo: batch_uniform stays in [0, 1] with expected mean",
          "[simd_mc]") {
    SIMDMonteCarlo mc(42);
    std::vector<float> out(4096, 0.0f);

    mc.batch_uniform(out.data(), static_cast<uint32_t>(out.size()));

    double sum = 0.0;
    for (float v : out) {
        REQUIRE(v >= 0.0f);
        REQUIRE(v <= 1.0f);
        sum += v;
    }

    const double mean = sum / static_cast<double>(out.size());
    REQUIRE(mean == Approx(0.5).margin(0.05));
}

TEST_CASE("SIMDMonteCarlo: batch_normal_clipped enforces clip floor",
          "[simd_mc]") {
    SIMDMonteCarlo mc(42);
    std::vector<float> out(2048, 0.0f);

    mc.batch_normal_clipped(
        out.data(),
        static_cast<uint32_t>(out.size()),
        -2.0f,
        1.0f,
        0.0f
    );

    const auto clipped = std::count(out.begin(), out.end(), 0.0f);
    for (float v : out) {
        REQUIRE(v >= 0.0f);
    }
    REQUIRE(clipped > 0);
}

TEST_CASE("SIMDMonteCarlo: batch_dirichlet rows are non-negative and sum to one",
          "[simd_mc]") {
    SIMDMonteCarlo mc(123);
    constexpr uint32_t n_draws = 128;
    constexpr uint32_t n_cats = 3;
    const float alphas[n_cats] = {2.0f, 3.0f, 1.0f};
    std::vector<float> out(n_draws * n_cats, 0.0f);

    mc.batch_dirichlet(out.data(), n_draws, alphas, n_cats);

    for (uint32_t draw = 0; draw < n_draws; ++draw) {
        float row_sum = 0.0f;
        for (uint32_t cat = 0; cat < n_cats; ++cat) {
            const float v = out[draw * n_cats + cat];
            REQUIRE(v >= 0.0f);
            row_sum += v;
        }
        REQUIRE(row_sum == Approx(1.0f).margin(1e-5));
    }
}

TEST_CASE("SIMDMonteCarlo: zero Cholesky factor returns means exactly",
          "[simd_mc]") {
    SIMDMonteCarlo mc(7);
    constexpr uint32_t n_draws = 16;
    constexpr uint32_t n_vars = 3;
    const float means[n_vars] = {1.5f, -2.0f, 7.25f};
    const float zero_chol[n_vars * n_vars] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    std::vector<float> out(n_draws * n_vars, 0.0f);

    mc.batch_multivariate_normal(
        out.data(),
        n_draws,
        means,
        zero_chol,
        n_vars
    );

    for (uint32_t draw = 0; draw < n_draws; ++draw) {
        for (uint32_t var = 0; var < n_vars; ++var) {
            REQUIRE(out[draw * n_vars + var] == Approx(means[var]).margin(1e-7));
        }
    }
}

// ---------------------------------------------------------------------------
// NEON Box-Muller statistical validation
//
// On ARM builds (Apple Silicon M1/M2/M3) batch_normal() dispatches to the
// NEON-vectorized neon_box_muller() path. On x86/scalar builds it uses the
// scalar box_muller_pair() path. This test validates the public output of
// batch_normal(mean=0, std=1) against known N(0,1) properties, catching any
// NEON codegen regression (e.g. wrong vld1q_f32 offset, bad vfmaq intrinsic).
// ---------------------------------------------------------------------------
TEST_CASE("SIMDMonteCarlo: batch_normal N(0,1) passes statistical checks",
          "[simd_mc][neon_validation]") {
    SIMDMonteCarlo mc(12345);
    constexpr uint32_t N = 10000;
    std::vector<float> out(N, 0.0f);

    mc.batch_normal(out.data(), N, 0.0f, 1.0f);

    // All samples must be finite.
    for (float v : out) {
        REQUIRE(std::isfinite(v));
    }

    // Mean ≈ 0 (within 3σ/√N ≈ 0.03 for N=10000).
    double sum = 0.0;
    for (float v : out) sum += v;
    const double mean = sum / N;
    REQUIRE(mean == Approx(0.0).margin(0.05));

    // Std ≈ 1 (within ≈5% at N=10000).
    double sq_sum = 0.0;
    for (float v : out) sq_sum += (v - mean) * (v - mean);
    const double std_dev = std::sqrt(sq_sum / N);
    REQUIRE(std_dev == Approx(1.0).margin(0.05));

    // 68-95-99.7 rule: empirical fractions within 1σ/2σ/3σ must match theory.
    int within1 = 0, within2 = 0, within3 = 0;
    for (float v : out) {
        const double av = std::abs(v - mean);
        if (av <= 1.0 * std_dev) ++within1;
        if (av <= 2.0 * std_dev) ++within2;
        if (av <= 3.0 * std_dev) ++within3;
    }
    // 68.27% ± 2% tolerance.
    REQUIRE(static_cast<double>(within1) / N == Approx(0.6827).margin(0.02));
    // 95.45% ± 1% tolerance.
    REQUIRE(static_cast<double>(within2) / N == Approx(0.9545).margin(0.01));
    // 99.73% ± 0.5% tolerance.
    REQUIRE(static_cast<double>(within3) / N == Approx(0.9973).margin(0.005));
}

TEST_CASE("SIMDMonteCarlo: batch_normal N(mu, sigma²) preserves location-scale",
          "[simd_mc][neon_validation]") {
    // Verify that mean and std parameters are applied correctly in the NEON path.
    SIMDMonteCarlo mc(99);
    constexpr uint32_t N = 8192;
    constexpr float TARGET_MEAN = 75.0f;
    constexpr float TARGET_STD  = 12.5f;
    std::vector<float> out(N, 0.0f);

    mc.batch_normal(out.data(), N, TARGET_MEAN, TARGET_STD);

    double sum = 0.0;
    for (float v : out) sum += v;
    const double mean = sum / N;
    REQUIRE(mean == Approx(static_cast<double>(TARGET_MEAN)).margin(0.5));

    double sq_sum = 0.0;
    for (float v : out) sq_sum += (v - mean) * (v - mean);
    const double std_dev = std::sqrt(sq_sum / N);
    REQUIRE(std_dev == Approx(static_cast<double>(TARGET_STD)).margin(0.5));
}

TEST_CASE("SIMDMonteCarlo C API: wrappers produce valid samples",
          "[simd_mc][c_api]") {
    void* handle = simd_mc_create(99);
    REQUIRE(handle != nullptr);

    std::vector<float> normal_out(256, 0.0f);
    simd_normal_clipped(handle, normal_out.data(), 256, -1.0f, 1.0f, 0.0f);
    for (float v : normal_out) {
        REQUIRE(v >= 0.0f);
        REQUIRE(std::isfinite(v));
    }

    const float alphas[2] = {1.0f, 1.0f};
    std::vector<float> dirichlet_out(64 * 2, 0.0f);
    simd_dirichlet_sample(handle, dirichlet_out.data(), 64, alphas, 2);
    for (size_t i = 0; i < dirichlet_out.size(); i += 2) {
        REQUIRE(dirichlet_out[i] >= 0.0f);
        REQUIRE(dirichlet_out[i + 1] >= 0.0f);
        REQUIRE(dirichlet_out[i] + dirichlet_out[i + 1]
                == Approx(1.0f).margin(1e-5));
    }

    simd_mc_destroy(handle);
}
