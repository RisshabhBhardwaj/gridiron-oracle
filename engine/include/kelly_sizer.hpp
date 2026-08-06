/**
 * engine/include/kelly_sizer.hpp
 *
 * Fractional Kelly criterion for bet sizing (CLAUDE.md §3 / Interview anchor §2).
 *
 * Kelly derivation (must be reproducible on a whiteboard):
 *
 *   Maximize E[log(W)] where W is wealth after one bet with fraction f wagered.
 *     W(win)  = 1 + b*f        (b = net profit per unit wagered)
 *     W(lose) = 1 - f
 *     E[log(W)] = p*log(1 + b*f) + q*log(1 - f)   where q = 1 - p
 *
 *   Taking the derivative with respect to f and setting to zero:
 *     d/df = p*b / (1 + b*f) - q / (1 - f) = 0
 *     => p*b*(1 - f) = q*(1 + b*f)
 *     => p*b - p*b*f = q + q*b*f
 *     => p*b - q = f*(p*b + q*b) = f*b*(p + q) = f*b
 *     => f* = (b*p - q) / b
 *
 *   where:  b = decimal_odds - 1   (net profit per dollar risked)
 *           p = win_probability
 *           q = 1 - p
 *
 * Why fractional Kelly (0.25×)? (CLAUDE.md §5)
 *   Full Kelly maximizes long-run log-wealth but exhibits extreme variance.
 *   At fraction k: log-growth ≈ k*(1 - k/2) * G_full.
 *   At k = 0.25: growth ≈ 0.25 * 0.875 * G_full ≈ 21.9% of G_full per unit
 *   variance (k² = 6.25%). This is the standard quant sports-betting compromise:
 *   ~94% of the achievable growth rate at half the volatility of full Kelly.
 */
#pragma once

#include <algorithm>
#include <cmath>

namespace gridiron {

struct KellyConfig {
    double kelly_fraction     = 0.25;   // fractional multiplier (CLAUDE.md §5)
    double max_bet_fraction   = 0.05;   // hard cap: never exceed 5% of bankroll
    double min_edge_threshold = 0.01;   // ignore bets with edge (f*) below 1%
};

class KellySizer {
public:
    explicit KellySizer(KellyConfig cfg = {}) noexcept : cfg_(cfg) {}

    /**
     * size_bet — Compute the recommended bet fraction of bankroll.
     *
     * @param win_prob     Estimated win probability p ∈ (0, 1)
     * @param decimal_odds Market decimal odds (e.g. 2.10 → net odds b = 1.10)
     * @return             Bet fraction ∈ [0, max_bet_fraction]
     *
     * Returns 0.0 if:
     *   - win_prob is not in (0, 1)
     *   - decimal_odds ≤ 1.0 (no positive net odds)
     *   - Full Kelly f* ≤ min_edge_threshold (bet has insufficient edge)
     */
    [[nodiscard]] double size_bet(double win_prob, double decimal_odds) const noexcept {
        if (win_prob <= 0.0 || win_prob >= 1.0) return 0.0;
        if (decimal_odds <= 1.0)                return 0.0;

        const double b = decimal_odds - 1.0;   // net profit per unit wagered
        const double p = win_prob;
        const double q = 1.0 - p;

        // Full Kelly fraction: f* = (b*p - q) / b
        const double f_star = (b * p - q) / b;

        // Below-threshold edge or negative expectation → no bet.
        if (f_star <= cfg_.min_edge_threshold) return 0.0;

        // Fractional Kelly: scale down; then hard-cap at max_bet_fraction.
        const double f_fractional = cfg_.kelly_fraction * f_star;
        return std::min(f_fractional, cfg_.max_bet_fraction);
    }

    /**
     * bet_size_dollars — Convert fraction to dollar amount.
     *
     * @param bankroll     Current total bankroll in dollars
     * @param win_prob     Win probability
     * @param decimal_odds Market decimal odds
     * @return             Dollar bet amount (≥ 0)
     */
    [[nodiscard]] double bet_size_dollars(double bankroll,
                                          double win_prob,
                                          double decimal_odds) const noexcept {
        return bankroll * size_bet(win_prob, decimal_odds);
    }

    const KellyConfig& config() const noexcept { return cfg_; }

private:
    KellyConfig cfg_;
};

}  // namespace gridiron
