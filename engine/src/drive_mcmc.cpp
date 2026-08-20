// engine/src/drive_mcmc.cpp
//
// Drive-Level MCMC — Implementation.
// See engine/include/drive_mcmc.hpp for API documentation.
//
// ╔══════════════════════════════════════════════════════════════════╗
// ║  PHASE 5 STUB — NOT PRODUCTION-READY                           ║
// ║  Drive/play-level Markov chain requires PBP pipeline            ║
// ║  (BLOCKED: formation data needs NGS/PFF).                       ║
// ║  Catch2 tests exist for default transitions, policy boundaries, ║
// ║  CSV loading, and C API behavior; keep that coverage before     ║
// ║  promoting to production.                                       ║
// ╚══════════════════════════════════════════════════════════════════╝

#include "drive_mcmc.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace gridiron {

// ── Constructor
// ───────────────────────────────────────────────────────────────

DriveMCMC::DriveMCMC(uint32_t n_simulations, uint32_t rng_seed)
    : n_simulations_(n_simulations), rng_(rng_seed), norm_dist_(0.0f, 1.0f),
      unif_dist_(0.0f, 1.0f) {
  load_default_transitions();
}

// ── Table Indexing
// ────────────────────────────────────────────────────────────

static uint32_t table_index(uint8_t field_pos, uint8_t down, uint8_t ytg) {
  uint32_t fp_bucket = std::min(static_cast<uint32_t>(field_pos / 10), 10u);
  uint32_t down_idx = std::min(static_cast<uint32_t>(down - 1), 3u);
  uint32_t ytg_bucket = std::min(static_cast<uint32_t>((ytg - 1) / 5), 5u);
  return fp_bucket * (4 * 6) + down_idx * 6 + ytg_bucket;
}

const PlayDistribution &DriveMCMC::lookup(const DriveState &s) const {
  uint32_t idx = table_index(s.field_pos, s.down, s.yards_to_go);
  return transitions_[idx];
}

// ── Load Default Transitions
// ──────────────────────────────────────────────────

void DriveMCMC::load_default_transitions() {
  // Fill all cells with NFL-average distributions (2018-2024 regular season).
  // Real values from nflreadr/PBP analysis:
  //   avg yards/play: 5.8 (all plays)
  //   std:            7.5
  //   P(turnover):    0.027   (~2.7% per play; includes INT + fumbles lost)
  //   P(penalty+):    0.08
  //   P(penalty-):    0.05
  //
  // Zone-specific adjustments:
  //   Red zone (fp >= 80):   mean_gain = 4.2, std=6.5 (compressed field)
  //   Mid-field (fp 40-60):  mean_gain = 6.1
  //   Own territory (fp<40): mean_gain = 5.5

  for (uint32_t fp = 0; fp < N_FP_BUCKETS; ++fp) {
    float fp_real = fp * 10.0f;
    float mean_gain = (fp_real >= 80) ? 4.2f : (fp_real >= 40) ? 6.1f : 5.5f;
    float gain_std = (fp_real >= 80) ? 6.5f : 7.5f;
    float p_turn = (fp_real >= 80) ? 0.035f : 0.027f; // higher in red zone

    for (uint32_t d = 0; d < N_DOWN; ++d) {
      // 3rd/4th down: more desperate, slightly higher std
      float d_std_factor = (d >= 2) ? 1.2f : 1.0f;

      for (uint32_t ytg = 0; ytg < N_YTG_BUCKETS; ++ytg) {
        uint32_t idx = fp * (N_DOWN * N_YTG_BUCKETS) + d * N_YTG_BUCKETS + ytg;
        transitions_[idx] = {
            mean_gain, gain_std * d_std_factor, p_turn,
            0.08f, // p_penalty_gain
            0.05f, // p_penalty_loss
        };
      }
    }
  }
}

// ── CSV Loader
// ────────────────────────────────────────────────────────────────

bool DriveMCMC::load_transitions_from_csv(const std::string &csv_path) {
  std::ifstream f(csv_path);
  if (!f.is_open())
    return false;

  std::string line;
  std::getline(f, line); // skip header

  uint32_t rows = 0;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    std::string tok;
    std::vector<float> vals;
    while (std::getline(ss, tok, ',')) {
      try {
        vals.push_back(std::stof(tok));
      } catch (...) {
        vals.push_back(0.0f);
      }
    }
    if (vals.size() < 8)
      continue;
    // CSV columns: fp_bucket, down, ytg_bucket, mean_gain, gain_std,
    //              p_turnover, p_penalty_gain, p_penalty_loss
    uint32_t fp = static_cast<uint32_t>(vals[0]);
    uint32_t dn = static_cast<uint32_t>(vals[1]);
    uint32_t ytg = static_cast<uint32_t>(vals[2]);
    if (fp >= N_FP_BUCKETS || dn >= N_DOWN || ytg >= N_YTG_BUCKETS)
      continue;

    uint32_t idx = fp * (N_DOWN * N_YTG_BUCKETS) + dn * N_YTG_BUCKETS + ytg;
    transitions_[idx] = {vals[3], vals[4], vals[5], vals[6], vals[7]};
    ++rows;
  }
  return rows > 0;
}

// ── 4th Down Decision
// ─────────────────────────────────────────────────────────

DriveOutcome DriveMCMC::fourth_down_decision(uint8_t field_pos,
                                             uint8_t ytg) const {
  // Simplified decision tree based on NFL average coaching tendencies.
  // Post-2021 analytics revolution adjustments applied.
  if (field_pos >= 65) {
    // Red zone 4th — attempt FG if ytg <= 5, else go for it
    return (ytg <= 5) ? DriveOutcome::FIELD_GOAL : DriveOutcome::TURNOVER_DOWNS;
  } else if (field_pos >= 45) {
    // Mid-field — go if ytg <= 2, else punt
    return (ytg <= 2) ? DriveOutcome::IN_PROGRESS : DriveOutcome::PUNT;
  } else {
    // Own territory — always punt
    return DriveOutcome::PUNT;
  }
}

// ── Single Drive Simulation
// ───────────────────────────────────────────────────

DriveState DriveMCMC::advance_one_play(const DriveState &s) {
  const PlayDistribution &dist = lookup(s);
  DriveState next = s;

  // Turnover?
  if (unif_dist_(rng_) < dist.p_turnover) {
    next.outcome = DriveOutcome::TURNOVER;
    return next;
  }

  // Penalty?
  float u_pen = unif_dist_(rng_);
  float yards_gained;
  if (u_pen < dist.p_penalty_gain) {
    yards_gained = 5.0f; // automatic 5 yards + first down
  } else if (u_pen < dist.p_penalty_gain + dist.p_penalty_loss) {
    yards_gained = -5.0f; // 5-yard loss
  } else {
    // Normal play: sample from N(mean_gain, gain_std)
    float z = norm_dist_(rng_);
    yards_gained = dist.mean_gain + dist.gain_std * z;
  }

  // Update field position
  int new_fp = static_cast<int>(s.field_pos) + static_cast<int>(yards_gained);
  new_fp = std::max(0, std::min(new_fp, 100));

  // Safety check
  if (new_fp <= safety_threshold_ && yards_gained < 0) {
    next.outcome = DriveOutcome::SAFETY;
    return next;
  }

  // Touchdown!
  if (new_fp >= 97) {
    next.field_pos = static_cast<uint8_t>(new_fp);
    next.outcome = DriveOutcome::TOUCHDOWN;
    return next;
  }

  next.field_pos = static_cast<uint8_t>(new_fp);

  // Down counter
  int ytg_remaining =
      static_cast<int>(s.yards_to_go) - static_cast<int>(yards_gained);
  if (ytg_remaining <= 0 || u_pen < dist.p_penalty_gain) {
    // First down!
    next.down = 1;
    next.yards_to_go = 10;
  } else {
    next.down = s.down + 1;
    next.yards_to_go = static_cast<uint8_t>(std::min(ytg_remaining, 30));
  }

  // 4th down decision
  if (next.down > 4) {
    next.outcome = DriveOutcome::TURNOVER_DOWNS;
  } else if (next.down == 4) {
    DriveOutcome decision =
        fourth_down_decision(next.field_pos, next.yards_to_go);
    if (decision != DriveOutcome::IN_PROGRESS) {
      next.outcome = decision;
    }
  }

  return next;
}

DriveState DriveMCMC::simulate_single_drive(const DriveState &start) {
  DriveState state = start;
  state.outcome = DriveOutcome::IN_PROGRESS;

  for (uint32_t play = 0; play < max_plays_; ++play) {
    if (state.is_terminal())
      break;
    state = advance_one_play(state);
  }
  if (!state.is_terminal()) {
    state.outcome = DriveOutcome::PUNT; // timeout — count as punt
  }
  return state;
}

// ── simulate_drive
// ────────────────────────────────────────────────────────────

DriveResult DriveMCMC::simulate_drive(const DriveState &start) {
  uint32_t n_td = 0, n_fg = 0, n_punt = 0, n_turn = 0;
  std::vector<float> yards_per_path;
  yards_per_path.reserve(n_simulations_);

  for (uint32_t s = 0; s < n_simulations_; ++s) {
    DriveState terminal = simulate_single_drive(start);

    switch (terminal.outcome) {
    case DriveOutcome::TOUCHDOWN:
      ++n_td;
      break;
    case DriveOutcome::FIELD_GOAL:
      ++n_fg;
      break;
    case DriveOutcome::PUNT:
      ++n_punt;
      break;
    case DriveOutcome::TURNOVER_DOWNS:
    case DriveOutcome::TURNOVER:
    case DriveOutcome::SAFETY:
      ++n_turn;
      break;
    default:
      break;
    }

    int net_yards = static_cast<int>(terminal.field_pos) -
                    static_cast<int>(start.field_pos);
    yards_per_path.push_back(static_cast<float>(net_yards));
  }

  float n = static_cast<float>(n_simulations_);
  float p_td = n_td / n;
  float p_fg = n_fg / n;
  float p_pnt = n_punt / n;
  float p_trn = n_turn / n;

  // Percentiles of net yards
  std::vector<float> sorted_y = yards_per_path;
  std::sort(sorted_y.begin(), sorted_y.end());
  float mean_y = 0;
  for (float y : sorted_y)
    mean_y += y;
  mean_y /= n;

  auto pct = [&](float p) {
    return sorted_y[static_cast<uint32_t>(p * (n_simulations_ - 1))];
  };

  return DriveResult{
      p_td,
      p_fg,
      p_pnt,
      p_trn,
      mean_y,
      pct(0.50f),
      pct(0.90f),
      18.0f,                     // placeholder: expected plays (from NFL avg)
      7.0f * p_td + 3.0f * p_fg, // expected points
      n_simulations_,
  };
}

// ── simulate_drives_batch
// ──────────────────────────────────────────────────────

void DriveMCMC::simulate_drives_batch(const DriveState *starts,
                                      uint32_t n_starts, DriveResult *results) {
  for (uint32_t i = 0; i < n_starts; ++i) {
    results[i] = simulate_drive(starts[i]);
  }
}

} // namespace gridiron

// ── C API
// ──────────────────────────────────────────────────────────────────────

extern "C" {

void *drive_mcmc_create(uint32_t n_simulations, uint32_t rng_seed) {
  return new gridiron::DriveMCMC(n_simulations, rng_seed);
}

void drive_mcmc_destroy(void *handle) {
  delete static_cast<gridiron::DriveMCMC *>(handle);
}

void drive_mcmc_load_defaults(void *handle) {
  static_cast<gridiron::DriveMCMC *>(handle)->load_default_transitions();
}

void drive_mcmc_simulate(void *handle, uint8_t field_pos, uint8_t down,
                         uint8_t yards_to_go, float *out_p_td, float *out_p_fg,
                         float *out_expected_yards, float *out_drive_value) {
  auto *sim = static_cast<gridiron::DriveMCMC *>(handle);
  gridiron::DriveState start{field_pos, down, yards_to_go,
                             gridiron::DriveOutcome::IN_PROGRESS};
  gridiron::DriveResult res = sim->simulate_drive(start);
  *out_p_td = res.p_touchdown;
  *out_p_fg = res.p_field_goal;
  *out_expected_yards = res.expected_yards;
  *out_drive_value = res.drive_value;
}

bool drive_mcmc_load_transitions_csv(void *handle, const char *csv_path) {
  if (!handle || !csv_path) {
    return false;
  }
  return static_cast<gridiron::DriveMCMC *>(handle)->load_transitions_from_csv(
      csv_path);
}

} // extern "C"
