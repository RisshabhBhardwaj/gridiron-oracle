// engine/include/drive_mcmc.hpp
//
// Drive-Level MCMC — Markov Chain simulation over NFL drive states,
// conditioned on game script (Phase 5).
//
// OVERVIEW
// --------
// Models an NFL drive as a Markov chain over states:
//
//     State = (field_position, down, yards_to_go, score_differential, quarter)
//
//     where:
//         field_position:     0 (own goal-line) → 100 (opponent goal-line)
//         down:                1, 2, 3, or 4
//         yards_to_go:         1–30 (capped at 30 for state space tractability)
//         score_differential:  posteam's margin, fixed for the life of one
//                              simulated drive (a drive doesn't span scoring
//                              plays by construction)
//         quarter:             1-4, fixed for the life of one simulated drive;
//                              OT is clamped to 4
//
// score_differential and quarter exist so game script — a leading team
// running more, especially late — emerges from the fitted transition table
// rather than being assumed. Each state now also carries p_pass: the
// learned probability a play at that state is a pass, exposed on
// DriveResult as expected_pass_rate so callers (and tests) can check the
// simulated mix against the real one, not just the fitted table.
//
// Terminal states:
//     TOUCHDOWN (field_pos >= 97)
//     TURNOVER_ON_DOWNS (4th down, gain < yards_to_go)
//     PUNT (4th down, opted to punt instead)
//     FIELD_GOAL_ATTEMPT (4th down, close enough for FG)
//     SAFETY (tackled in own end zone)
//     TURNOVER (fumble/INT)
//
// Transition probabilities are learned from pbp_plays (Phase 2/5;
// ml/markov_simulator.py, scripts/export_drive_transitions.py). Without a
// loaded CSV, load_default_transitions() falls back to NFL-average priors
// with a documented, modest score/quarter skew (see drive_mcmc.cpp) — not a
// fit, just a plausible prior so an unconfigured engine doesn't behave
// nonsensically.
//
// USAGE
// -----
//   #include "drive_mcmc.hpp"
//   using namespace gridiron;
//
//   DriveMCMC sim(n_simulations=50000, rng_seed=42);
//   sim.load_transitions_from_csv("ml/oof/transitions_by_game_state.csv");
//
//   DriveResult res = sim.simulate_drive({25, 1, 10, 0, 1, DriveOutcome::IN_PROGRESS});
//   printf("P(TD) = %.2f, E[yards] = %.1f, pass rate = %.2f\n",
//          res.p_touchdown, res.expected_yards, res.expected_pass_rate);

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace gridiron {

// ── Drive State
// ───────────────────────────────────────────────────────────────

enum class DriveOutcome : uint8_t {
  IN_PROGRESS = 0,
  TOUCHDOWN = 1,
  FIELD_GOAL = 2,
  PUNT = 3,
  TURNOVER_DOWNS = 4,
  TURNOVER = 5,
  SAFETY = 6,
};

struct DriveState {
  uint8_t field_pos;   ///< 0 = own goal, 100 = opp goal. Drive ends at 97+ (TD)
  uint8_t down;        ///< 1-4
  uint8_t yards_to_go; ///< 1-30 (capped)
  int8_t score_differential = 0; ///< posteam margin, fixed for the whole drive
  uint8_t quarter = 1;            ///< 1-4 (OT clamps to 4), fixed for the whole drive
  DriveOutcome outcome; ///< IN_PROGRESS while drive is active

  bool is_terminal() const { return outcome != DriveOutcome::IN_PROGRESS; }
};

/// Summary statistics for one simulated drive (or distribution over many).
struct DriveResult {
  float p_touchdown;  ///< fraction of paths ending in TD
  float p_field_goal; ///< fraction ending in FG attempt
  float p_punt;       ///< fraction ending in punt
  float p_turnover;   ///< fraction ending in turnover (downs or fumble/INT)

  float expected_yards; ///< mean net yards gained over drive
  float p50_yards;      ///< median yards
  float p90_yards;      ///< 90th percentile yards

  float expected_plays;     ///< mean plays per drive
  float expected_pass_rate; ///< mean p_pass looked up across every play, every path
  float drive_value;        ///< expected points ≈ 7×P(TD) + 3×P(FG)

  uint32_t n_simulations;
};

// ── Play Distribution
// ─────────────────────────────────────────────────────────

/// Per-state play outcome distribution (learned from PBP data).
struct PlayDistribution {
  float mean_gain;      ///< mean yards per play
  float gain_std;       ///< std of yards gained
  float p_turnover;     ///< P(fumble or INT on this play)
  float p_penalty_gain; ///< P(penalty in favor of offense)
  float p_penalty_loss; ///< P(penalty against offense)
  float p_pass;         ///< P(play call is a pass) — informational, doesn't split the yardage model
};

// ── DriveMCMC ────────────────────────────────────────────────────────────────

class DriveMCMC {
public:
  explicit DriveMCMC(uint32_t n_simulations = 50000, uint32_t rng_seed = 42);
  ~DriveMCMC() = default;

  // ── Setup ─────────────────────────────────────────────────────────────────

  /// Load NFL-average transition distributions (no PBP data required).
  /// These approximate 2018-2024 NFL averages.
  void load_default_transitions();

  /// Load learned transitions from serialized CSV (Phase 5 PBP data,
  /// ml/markov_simulator.py's transitions_by_game_state export).
  /// CSV format: fp_bucket, down_idx, ytg_bucket, score_diff_bucket,
  ///             quarter_idx, mean_gain, gain_std, p_turnover,
  ///             p_penalty_gain, p_penalty_loss, p_pass, count
  bool load_transitions_from_csv(const std::string &csv_path);

  // ── Simulation ────────────────────────────────────────────────────────────

  /// Simulate n_simulations drives from the given starting state.
  /// @param start  Initial drive state
  /// @return       DriveResult with P distributions over outcomes
  DriveResult simulate_drive(const DriveState &start);

  /// Simulate drives from multiple starting positions in one call.
  /// @param starts   Array of starting states
  /// @param n_starts Number of starting states
  /// @param results  Pre-allocated array for output
  void simulate_drives_batch(const DriveState *starts, uint32_t n_starts,
                             DriveResult *results);

  // ── 4th Down Decision ──────────────────────────────────────────────────────

  /// Decision policy for 4th down: go / punt / field_goal.
  /// Returns next outcome based on field position and yards to go.
  DriveOutcome fourth_down_decision(uint8_t field_pos,
                                    uint8_t yards_to_go) const;

  // ── Configuration ─────────────────────────────────────────────────────────

  /// Maximum plays per drive before declaring it over (prevents infinite
  /// loops).
  uint32_t max_plays_per_drive() const { return max_plays_; }
  void set_max_plays(uint32_t n) { max_plays_ = n; }

  /// Field position threshold for TD zone (player sacked → safety).
  void set_safety_threshold(uint8_t v) { safety_threshold_ = v; }

private:
  uint32_t n_simulations_;
  uint32_t max_plays_ = 25;      // NFL drives rarely exceed 20 plays
  uint8_t safety_threshold_ = 3; // field_pos <= 3 → possible safety

  std::mt19937 rng_;
  std::normal_distribution<float> norm_dist_;
  std::uniform_real_distribution<float> unif_dist_;

  // Transition table: (fp_bucket, down, ytg_bucket, score_diff_bucket,
  // quarter_idx) → PlayDistribution. Buckets: field_pos /10 → 11 buckets,
  // ytg /5 → 6 buckets, score_differential → 5 buckets (see
  // score_diff_bucket in drive_mcmc.cpp — mirrors
  // ml.markov_simulator._score_diff_bucket), quarter 1-4 → 4 buckets.
  static constexpr uint32_t N_FP_BUCKETS = 11;
  static constexpr uint32_t N_DOWN = 4;
  static constexpr uint32_t N_YTG_BUCKETS = 6;
  static constexpr uint32_t N_SCORE_BUCKETS = 5;
  static constexpr uint32_t N_QUARTER = 4;
  static constexpr uint32_t TABLE_SIZE =
      N_FP_BUCKETS * N_DOWN * N_YTG_BUCKETS * N_SCORE_BUCKETS * N_QUARTER;

  std::array<PlayDistribution, TABLE_SIZE> transitions_;

  // Look up the play distribution for a given state.
  const PlayDistribution &lookup(const DriveState &s) const;

  // Simulate one play from state s, return the new state. Accumulates the
  // looked-up p_pass and a play count into the given running totals so
  // simulate_drive can report expected_pass_rate.
  DriveState advance_one_play(const DriveState &s, float &pass_sum, uint32_t &play_count);

  // Simulate one complete drive path from start, return terminal state.
  // Accumulates into pass_sum/play_count same as advance_one_play.
  DriveState simulate_single_drive(const DriveState &start, float &pass_sum, uint32_t &play_count);
};

} // namespace gridiron

// ── C API
// ─────────────────────────────────────────────────────────────────────
//
// Ownership contract:
//   - drive_mcmc_create() allocates a DriveMCMC on the heap and returns
//     an opaque void* handle. The caller owns this allocation.
//   - drive_mcmc_destroy() frees the allocation. The caller MUST call this
//     exactly once for every successful drive_mcmc_create() call to avoid
//     a memory leak. Passing NULL is safe (no-op).
//   - All other functions require a non-NULL handle from drive_mcmc_create().
//     Passing a destroyed or NULL handle is undefined behaviour.
//
// Thread safety: each handle is NOT thread-safe. Use one handle per thread
// or protect with an external mutex.

extern "C" {

void *drive_mcmc_create(uint32_t n_simulations, uint32_t rng_seed);
void drive_mcmc_destroy(void *handle);
void drive_mcmc_load_defaults(void *handle);

/// score_differential/quarter select the fitted state alongside field_pos/
/// down/yards_to_go (see DriveState); out_pass_rate is the simulated mean
/// P(pass) across every play, every simulated path from this start.
void drive_mcmc_simulate(void *handle, uint8_t field_pos, uint8_t down,
                         uint8_t yards_to_go, int8_t score_differential,
                         uint8_t quarter, float *out_p_td, float *out_p_fg,
                         float *out_expected_yards, float *out_pass_rate,
                         float *out_drive_value);
bool drive_mcmc_load_transitions_csv(void *handle, const char *csv_path);

} // extern "C"
