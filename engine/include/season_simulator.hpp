// engine/include/season_simulator.hpp
//
// C++ Season Simulator — High-Speed Monte Carlo Season Projection.
//
// OVERVIEW
// --------
// This is a C++ port of ml/season_simulator.py, designed to run 10,000+
// simulation paths in milliseconds (vs. ~30 seconds for pure Python).
//
// The simulator runs a week-by-week autoregressive loop:
//   For each week w in [start_week, end_week]:
//     1. Draw correlated samples from each player's Gaussian prior
//        using the Cholesky decomposition of the correlation matrix.
//     2. Clip at 0 (stats are non-negative).
//     3. Update Kalman posterior: x_k = x_{k-1} + K × (simulated - x_{k-1})
//     4. Accumulate season totals.
//
// Python integration: compile as shared library, bind via ctypes:
//   sim_week():   simulate one week, return mean/p10/p50/p90
//   sim_season(): simulate full season, return accumulated totals
//
// COMPILATION
// -----------
//   cd engine && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
//   Produces: engine/build/libgridiron_engine.dylib
//
// PYTHON USAGE
// ------------
//   import ctypes, numpy as np
//   lib = ctypes.CDLL("engine/build/libgridiron_engine.dylib")
//   # See extern "C" API at bottom of season_simulator.cpp

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <functional>
#include <memory>

namespace gridiron {

// ── Constants ─────────────────────────────────────────────────────────────────

constexpr float  KALMAN_GAIN      = 0.30f;   // mirrors Python KALMAN_GAIN
constexpr int    MAX_PLAYERS      = 512;     // max players per simulation batch
constexpr int    MAX_SIMULATIONS  = 100000;  // hard cap on Monte Carlo paths
constexpr int    NFL_WEEKS        = 18;      // regular season weeks

// ── Data Structures ───────────────────────────────────────────────────────────

/// Per-player projection prior — inputs to the simulator.
struct PlayerProjection {
    char     player_id[32];      ///< null-terminated UTF-8 player ID
    char     position[4];        ///< QB/RB/WR/TE
    char     team[4];            ///< 2-3 char team abbreviation

    float    kalman_mean;        ///< Kalman posterior mean for target stat
    float    kalman_variance;    ///< Kalman posterior variance (sigma²)

    float    injury_multiplier;  ///< 0.0 = out, 1.0 = healthy, 0.5 = limited
    float    snap_share;         ///< [0, 1] snap participation rate
};

/// Simulation output for one player across all simulation paths.
struct PlayerSimResult {
    char     player_id[32];
    uint32_t n_simulations;
    float    mean;    ///< mean of season totals across all paths
    float    std;     ///< standard deviation
    float    p10;     ///< 10th percentile
    float    p25;     ///< 25th percentile
    float    p50;     ///< median
    float    p75;     ///< 75th percentile
    float    p90;     ///< 90th percentile
};

/// Full season simulation output.
struct SeasonSimOutput {
    uint32_t          n_players;
    uint32_t          n_simulations;
    uint32_t          n_weeks;
    PlayerSimResult*  results;      ///< caller-owned array of n_players results
};

// ── SeasonSimulator Class ─────────────────────────────────────────────────────

class SeasonSimulator {
public:
    SeasonSimulator(uint32_t n_simulations = 10000, uint32_t rng_seed = 42);
    ~SeasonSimulator() = default;

    // ── Core simulation ────────────────────────────────────────────────────────

    /// Simulate one week. Returns n_players simulation draws.
    /// @param players    Array of PlayerProjection (n_players entries)
    /// @param n_players  Length of players array
    /// @param corr_mat   Flattened correlation matrix (n_players × n_players, row-major)
    ///                   Pass nullptr to use independent sampling.
    /// @param out        Pre-allocated output array (n_simulations × n_players, row-major)
    void sim_week(
        const PlayerProjection* players,
        uint32_t                n_players,
        const float*            corr_mat,   // [n_players × n_players] or nullptr
        float*                  out         // [n_simulations × n_players]
    );

    /// Simulate a full season [start_week, end_week].
    /// Applies Kalman update after each week (autoregressive).
    ///
    /// @param players      Initial player projections
    /// @param n_players    Number of players
    /// @param corr_mat     Correlation matrix (nullable for independence)
    /// @param start_week   First week to simulate (1-indexed)
    /// @param end_week     Last week to simulate (1-indexed, inclusive)
    /// @param output       Pre-allocated SeasonSimOutput (caller provides results array)
    void sim_season(
        const PlayerProjection* players,
        uint32_t                n_players,
        const float*            corr_mat,
        uint32_t                start_week,
        uint32_t                end_week,
        SeasonSimOutput*        output
    );

    // ── Percentile computation ──────────────────────────────────────────────────

    /// Compute [mean, std, p10, p25, p50, p75, p90] from a sample array.
    static void compute_percentiles(
        const float* samples,
        uint32_t     n,
        float*       out_stats  // [7] = {mean, std, p10, p25, p50, p75, p90}
    );

private:
    uint32_t                     n_simulations_;
    std::mt19937                 rng_;
    std::normal_distribution<float> normal_dist_;

    // Cholesky decomposition cache (player count → L matrix)
    std::vector<float>           last_cholesky_;
    uint32_t                     last_n_players_ = 0;

    // Apply Cholesky decomposition in-place to a correlation matrix.
    // Result stored in last_cholesky_. Called before each sim_week.
    bool cholesky_decompose(const float* corr_mat, uint32_t n);

    // Draw n_simulations correlated standard normal samples using last_cholesky_.
    // out shape: [n_simulations × n_players], row-major.
    void draw_correlated_normals(float* out, uint32_t n_players);

    // Apply Kalman update to player means given simulated week outcomes.
    static void kalman_update(
        PlayerProjection* players,
        uint32_t          n_players,
        const float*      week_outcomes  // shape [n_players], column means of week samples
    );
};

} // namespace gridiron


// ── C API for Python ctypes binding ───────────────────────────────────────────
// Compiled as extern "C" in season_simulator.cpp

extern "C" {

/// Create a new SeasonSimulator instance.
void* season_sim_create(uint32_t n_simulations, uint32_t rng_seed);

/// Free a SeasonSimulator instance.
void season_sim_destroy(void* handle);

/// Run a one-week simulation.
/// @param handle       Pointer from season_sim_create()
/// @param player_data  Flattened array of PlayerProjection structs
/// @param n_players    Number of players
/// @param corr_mat     Correlation matrix [n_players × n_players] or NULL
/// @param out          Output [n_simulations × n_players] float32 row-major
void season_sim_week(
    void*          handle,
    const float*   player_data,
    uint32_t       n_players,
    const float*   corr_mat,
    float*         out
);

/// Run a full-season simulation.
/// @param start_week   1-indexed first week
/// @param end_week     1-indexed last week (inclusive)
/// @param out_means    (n_players,) output mean season totals
/// @param out_p10      (n_players,) output p10 season totals
/// @param out_p50      (n_players,) output p50 season totals
/// @param out_p90      (n_players,) output p90 season totals
void season_sim_full(
    void*          handle,
    const float*   player_data,
    uint32_t       n_players,
    const float*   corr_mat,
    uint32_t       start_week,
    uint32_t       end_week,
    float*         out_means,
    float*         out_p10,
    float*         out_p50,
    float*         out_p90
);

} // extern "C"
