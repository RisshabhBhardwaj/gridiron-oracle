// engine/src/season_simulator.cpp
//
// Implementation of the C++ Season Simulator.
// See engine/include/season_simulator.hpp for API documentation.

#include "season_simulator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace gridiron {

// ── SeasonSimulator
// ────────────────────────────────────────────────────────────

SeasonSimulator::SeasonSimulator(uint32_t n_simulations, uint32_t rng_seed)
    : n_simulations_(
          std::min(n_simulations, static_cast<uint32_t>(MAX_SIMULATIONS))),
      rng_(rng_seed), normal_dist_(0.0f, 1.0f) {}

// ── Cholesky Decomposition
// ─────────────────────────────────────────────────────

bool SeasonSimulator::cholesky_decompose(const float *corr_mat, uint32_t n) {
  // In-place lower Cholesky via Cholesky–Banachiewicz.
  // Result stored in last_cholesky_ (n×n, row-major).
  last_cholesky_.assign(n * n, 0.0f);
  last_n_players_ = n;

  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t j = 0; j <= i; ++j) {
      float sum = corr_mat[i * n + j];
      for (uint32_t k = 0; k < j; ++k) {
        sum -= last_cholesky_[i * n + k] * last_cholesky_[j * n + k];
      }
      if (i == j) {
        if (sum <= 0.0f) {
          // Matrix not positive definite — add jitter and retry
          sum = 1e-6f;
        }
        last_cholesky_[i * n + j] = std::sqrt(sum);
      } else {
        last_cholesky_[i * n + j] = sum / last_cholesky_[j * n + j];
      }
    }
  }
  return true;
}

// ── Correlated Normal Draws
// ────────────────────────────────────────────────────

void SeasonSimulator::draw_correlated_normals(float *out, uint32_t n_players) {
  // Draw n_simulations × n_players standard normals, then apply Cholesky.
  // out shape: [n_simulations × n_players], row-major.
  const uint32_t ns = n_simulations_;
  const float *L = last_cholesky_.data();

  // Draw iid standard normals Z
  std::vector<float> Z(ns * n_players);
  for (float &v : Z) {
    v = normal_dist_(rng_);
  }

  // Apply L: Y = L × Z^T  →  out[s, i] = Σ_k L[i,k] × Z[s,k]
  for (uint32_t s = 0; s < ns; ++s) {
    for (uint32_t i = 0; i < n_players; ++i) {
      float acc = 0.0f;
      for (uint32_t k = 0; k <= i; ++k) {
        acc += L[i * n_players + k] * Z[s * n_players + k];
      }
      out[s * n_players + i] = acc;
    }
  }
}

// ── Percentile Computation
// ─────────────────────────────────────────────────────

void SeasonSimulator::compute_percentiles(const float *samples, uint32_t n,
                                          float *out_stats) {
  if (n == 0) {
    for (int i = 0; i < 7; ++i)
      out_stats[i] = 0.0f;
    return;
  }

  // Copy and sort
  std::vector<float> v(samples, samples + n);
  std::sort(v.begin(), v.end());

  // Mean
  double sum = 0.0;
  for (float x : v)
    sum += x;
  float mean = static_cast<float>(sum / n);

  // Std
  double sq_sum = 0.0;
  for (float x : v)
    sq_sum += (x - mean) * (x - mean);
  float std_val = static_cast<float>(std::sqrt(sq_sum / n));

  // Percentiles via linear interpolation
  auto percentile = [&](float p) -> float {
    float idx = p * (n - 1);
    uint32_t lo = static_cast<uint32_t>(idx);
    uint32_t hi = std::min(lo + 1, n - 1);
    float frac = idx - lo;
    return v[lo] + frac * (v[hi] - v[lo]);
  };

  out_stats[0] = mean;
  out_stats[1] = std_val;
  out_stats[2] = percentile(0.10f);
  out_stats[3] = percentile(0.25f);
  out_stats[4] = percentile(0.50f);
  out_stats[5] = percentile(0.75f);
  out_stats[6] = percentile(0.90f);
}

// ── Kalman Update
// ─────────────────────────────────────────────────────────────

void SeasonSimulator::kalman_update(PlayerProjection *players,
                                    uint32_t n_players,
                                    const float *week_outcomes) {
  for (uint32_t i = 0; i < n_players; ++i) {
    float prior = players[i].kalman_mean;
    float y = week_outcomes[i];
    // x_k = x_{k-1} + K × (y - x_{k-1})
    players[i].kalman_mean = prior + KALMAN_GAIN * (y - prior);
  }
}

// ── sim_week
// ──────────────────────────────────────────────────────────────────

void SeasonSimulator::sim_week(const PlayerProjection *players,
                               uint32_t n_players, const float *corr_mat,
                               float *out) {
  const uint32_t ns = n_simulations_;

  if (corr_mat != nullptr && n_players > 1) {
    // Correlated sampling via Cholesky
    if (last_n_players_ != n_players) {
      cholesky_decompose(corr_mat, n_players);
    }
    draw_correlated_normals(out, n_players);

    // Scale by per-player std and add mean; clip at 0
    for (uint32_t s = 0; s < ns; ++s) {
      for (uint32_t i = 0; i < n_players; ++i) {
        float sigma = std::sqrt(std::max(players[i].kalman_variance, 1e-6f));
        float raw = players[i].kalman_mean + sigma * out[s * n_players + i];
        // Apply injury multiplier and snap share
        raw *= players[i].injury_multiplier * players[i].snap_share;
        out[s * n_players + i] = std::max(raw, 0.0f);
      }
    }
  } else {
    // Independent sampling
    std::normal_distribution<float> norm(0.0f, 1.0f);
    for (uint32_t s = 0; s < ns; ++s) {
      for (uint32_t i = 0; i < n_players; ++i) {
        float sigma = std::sqrt(std::max(players[i].kalman_variance, 1e-6f));
        float raw = players[i].kalman_mean + sigma * norm(rng_);
        raw *= players[i].injury_multiplier * players[i].snap_share;
        out[s * n_players + i] = std::max(raw, 0.0f);
      }
    }
  }
}

// ── sim_season
// ────────────────────────────────────────────────────────────────

void SeasonSimulator::sim_season(const PlayerProjection *players_in,
                                 uint32_t n_players, const float *corr_mat,
                                 uint32_t start_week, uint32_t end_week,
                                 SeasonSimOutput *output) {
  assert(end_week >= start_week);
  assert(n_players <= MAX_PLAYERS);

  const uint32_t ns = n_simulations_;
  const uint32_t n_weeks = end_week - start_week + 1;

  // Make mutable copies of player projections for Kalman updates
  std::vector<PlayerProjection> players(players_in, players_in + n_players);

  // season_totals[sim × player] — accumulated over all weeks
  std::vector<float> season_totals(ns * n_players, 0.0f);
  // Temp buffer for one week's draws
  std::vector<float> week_buf(ns * n_players);

  // Initialize Cholesky if correlated
  if (corr_mat && n_players > 1) {
    cholesky_decompose(corr_mat, n_players);
  }

  // Extract unique teams and map player -> team_idx
  std::unordered_map<std::string, uint32_t> team_to_idx;
  std::vector<uint32_t> player_team_idx(n_players, 0);
  uint32_t n_teams = 0;
  for (uint32_t i = 0; i < n_players; ++i) {
    std::string t(players[i].team);
    if (t.empty())
      continue;
    if (team_to_idx.find(t) == team_to_idx.end()) {
      team_to_idx[t] = n_teams++;
    }
    player_team_idx[i] = team_to_idx[t];
  }

  // Dynamic Team Elo per simulation path: [ns * n_teams]
  std::vector<float> sim_off_elo(ns * n_teams, 1500.0f);

  for (uint32_t w = 0; w < n_weeks; ++w) {
    sim_week(players.data(), n_players, corr_mat, week_buf.data());

    // Track team point proxies and update dynamic Elo
    for (uint32_t s = 0; s < ns; ++s) {
      std::vector<float> team_week_pts(n_teams, 0.0f);

      for (uint32_t i = 0; i < n_players; ++i) {
        uint32_t tidx = player_team_idx[i];

        // Apply dynamic Team Elo modifier to the player's simulated week out
        // Baseline 1500 Elo. 100 points = ~10% bump.
        float elo_mod =
            1.0f + ((sim_off_elo[s * n_teams + tidx] - 1500.0f) * 0.001f);
        week_buf[s * n_players + i] *= std::max(elo_mod, 0.5f);

        team_week_pts[tidx] += week_buf[s * n_players + i];
        season_totals[s * n_players + i] += week_buf[s * n_players + i];
      }

      // Dynamically adjust Team Elo inside this specific simulation path
      // Proxy: If team total fantasy points < 40, they did terribly
      // (injury cascade / bad script). Drop Elo instantly by 25.
      // If > 100, they did great, bump Elo by 10.
      for (uint32_t t = 0; t < n_teams; ++t) {
        if (team_week_pts[t] < 40.0f) {
          sim_off_elo[s * n_teams + t] -= 25.0f; // Injury / tank
        } else if (team_week_pts[t] > 100.0f) {
          sim_off_elo[s * n_teams + t] += 10.0f; // Breakout script
        }
      }
    }

    // Kalman update: compute column means of this week's draws
    std::vector<float> week_means(n_players, 0.0f);
    for (uint32_t s = 0; s < ns; ++s) {
      for (uint32_t i = 0; i < n_players; ++i) {
        week_means[i] += week_buf[s * n_players + i];
      }
    }
    for (float &v : week_means)
      v /= ns;

    kalman_update(players.data(), n_players, week_means.data());
  }

  // Write percentile summaries into output
  output->n_players = n_players;
  output->n_simulations = ns;
  output->n_weeks = n_weeks;

  std::vector<float> player_samples(ns);
  for (uint32_t i = 0; i < n_players; ++i) {
    // Collect all simulation paths for player i
    for (uint32_t s = 0; s < ns; ++s) {
      player_samples[s] = season_totals[s * n_players + i];
    }

    float stats[7];
    compute_percentiles(player_samples.data(), ns, stats);

    PlayerSimResult &r = output->results[i];
    std::strncpy(r.player_id, players_in[i].player_id, 31);
    r.player_id[31] = '\0';
    r.n_simulations = ns;
    r.mean = stats[0];
    r.std = stats[1];
    r.p10 = stats[2];
    r.p25 = stats[3];
    r.p50 = stats[4];
    r.p75 = stats[5];
    r.p90 = stats[6];
  }
}

} // namespace gridiron

// ── C API
// ──────────────────────────────────────────────────────────────────────

extern "C" {

void *season_sim_create(uint32_t n_simulations, uint32_t rng_seed) {
  return new gridiron::SeasonSimulator(n_simulations, rng_seed);
}

void season_sim_destroy(void *handle) {
  delete static_cast<gridiron::SeasonSimulator *>(handle);
}

void season_sim_week(void *handle, const float *player_data, uint32_t n_players,
                     const float *corr_mat, float *out) {
  auto *sim = static_cast<gridiron::SeasonSimulator *>(handle);
  auto *players =
      reinterpret_cast<const gridiron::PlayerProjection *>(player_data);
  sim->sim_week(players, n_players, corr_mat, out);
}

void season_sim_full(void *handle, const float *player_data, uint32_t n_players,
                     const float *corr_mat, uint32_t start_week,
                     uint32_t end_week, float *out_means, float *out_p10,
                     float *out_p50, float *out_p90) {
  auto *sim = static_cast<gridiron::SeasonSimulator *>(handle);
  auto *players_in =
      reinterpret_cast<const gridiron::PlayerProjection *>(player_data);

  // Allocate result storage on the stack (n_players capped at MAX_PLAYERS)
  std::vector<gridiron::PlayerSimResult> results(n_players);
  gridiron::SeasonSimOutput output{};
  output.results = results.data();

  sim->sim_season(players_in, n_players, corr_mat, start_week, end_week,
                  &output);

  for (uint32_t i = 0; i < n_players; ++i) {
    out_means[i] = results[i].mean;
    out_p10[i] = results[i].p10;
    out_p50[i] = results[i].p50;
    out_p90[i] = results[i].p90;
  }
}

} // extern "C"
