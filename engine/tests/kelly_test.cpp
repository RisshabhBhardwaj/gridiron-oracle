/**
 * engine/tests/kelly_test.cpp
 *
 * Catch2 v3 unit tests for KellySizer.
 *
 * All expected values verified by hand using:
 *   f* = (b*p - q) / b,   b = decimal_odds - 1,   q = 1 - p
 *   fractional = kelly_fraction * f*
 *   result = min(fractional, max_bet_fraction)
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kelly_sizer.hpp"

using namespace gridiron;
using Catch::Approx;

TEST_CASE("KellySizer: known value — result capped at max_bet_fraction", "[kelly]") {
    // p=0.60, decimal_odds=2.10, b=1.10
    // f* = (1.10*0.60 - 0.40) / 1.10 = (0.66 - 0.40) / 1.10 = 0.26/1.10 ≈ 0.23636
    // fractional = 0.25 * 0.23636 ≈ 0.05909  >  max_bet_fraction=0.05
    // capped → 0.05
    KellySizer sizer;
    REQUIRE(sizer.size_bet(0.60, 2.10) == Approx(0.05).epsilon(1e-9));
}

TEST_CASE("KellySizer: positive edge below cap (custom config)", "[kelly]") {
    // With kelly_fraction=0.10 and max_bet_fraction=0.10 the cap doesn't bind:
    // p=0.60, b=1.10: f* ≈ 0.23636
    // fractional = 0.10 * 0.23636 ≈ 0.023636  <  0.10 cap  → uncapped
    KellyConfig cfg;
    cfg.kelly_fraction   = 0.10;
    cfg.max_bet_fraction = 0.10;
    KellySizer sizer(cfg);

    const double expected = 0.10 * (1.10 * 0.60 - 0.40) / 1.10;
    REQUIRE(sizer.size_bet(0.60, 2.10) == Approx(expected).epsilon(1e-9));
}

TEST_CASE("KellySizer: negative edge returns zero", "[kelly]") {
    // p=0.40, decimal_odds=2.0, b=1.0
    // f* = (1.0*0.40 - 0.60) / 1.0 = -0.20  → negative, clamped to 0
    KellySizer sizer;
    REQUIRE(sizer.size_bet(0.40, 2.0) == Approx(0.0).epsilon(1e-9));
}

TEST_CASE("KellySizer: zero probability returns zero", "[kelly]") {
    KellySizer sizer;
    REQUIRE(sizer.size_bet(0.0, 2.0) == Approx(0.0).epsilon(1e-9));
}

TEST_CASE("KellySizer: probability >= 1.0 returns zero", "[kelly]") {
    KellySizer sizer;
    REQUIRE(sizer.size_bet(1.0,  2.0) == Approx(0.0).epsilon(1e-9));
    REQUIRE(sizer.size_bet(1.05, 2.0) == Approx(0.0).epsilon(1e-9));
}

TEST_CASE("KellySizer: decimal_odds <= 1.0 returns zero (no net profit)", "[kelly]") {
    KellySizer sizer;
    REQUIRE(sizer.size_bet(0.70, 1.0) == Approx(0.0).epsilon(1e-9));
    REQUIRE(sizer.size_bet(0.70, 0.5) == Approx(0.0).epsilon(1e-9));
}

TEST_CASE("KellySizer: hard limit always enforced", "[kelly]") {
    // Even near-certainty probability must not exceed max_bet_fraction.
    KellySizer sizer;
    REQUIRE(sizer.size_bet(0.99, 2.0) <= sizer.config().max_bet_fraction + 1e-9);
    REQUIRE(sizer.size_bet(0.95, 3.0) <= sizer.config().max_bet_fraction + 1e-9);
}

TEST_CASE("KellySizer: edge below min_edge_threshold returns zero", "[kelly]") {
    // p=0.502, decimal_odds=2.0, b=1.0
    // f* = (1.0*0.502 - 0.498) / 1.0 = 0.004  < min_edge_threshold=0.01  → 0
    KellySizer sizer;
    REQUIRE(sizer.size_bet(0.502, 2.0) == Approx(0.0).epsilon(1e-9));
}

TEST_CASE("KellySizer: bet_size_dollars = bankroll × size_bet", "[kelly]") {
    // size_bet(0.60, 2.10) == 0.05 (verified above)
    // 1000 × 0.05 = 50.0
    KellySizer sizer;
    REQUIRE(sizer.bet_size_dollars(1000.0, 0.60, 2.10) == Approx(50.0).epsilon(1e-6));
}

TEST_CASE("KellySizer: zero bankroll produces zero dollar bet", "[kelly]") {
    KellySizer sizer;
    REQUIRE(sizer.bet_size_dollars(0.0, 0.60, 2.10) == Approx(0.0).epsilon(1e-9));
}
