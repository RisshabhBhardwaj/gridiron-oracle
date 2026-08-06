/**
 * engine/tests/exposure_manager_test.cpp
 *
 * Catch2 v3 unit tests for ExposureManager.
 * Verifies per-player, per-game, and total in-play limit enforcement,
 * plus release_bet / release_game cleanup.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "exposure_manager.hpp"

using namespace gridiron;
using Catch::Approx;

// Standard test config: bankroll=$1000, player=5%, game=20%, total=20%
static ExposureConfig test_config() {
    ExposureConfig cfg;
    cfg.bankroll                   = 1000.0;
    cfg.max_per_player_fraction    = 0.05;
    cfg.max_per_game_fraction      = 0.20;
    cfg.max_total_in_play_fraction = 0.20;
    return cfg;
}

TEST_CASE("ExposureManager: single bet within all limits is approved", "[exposure]") {
    ExposureManager em(test_config());
    BetRecord bet{"p1", "game1", "WR", 30.0};  // 3% of $1000 — within 5% player cap

    REQUIRE(em.try_place_bet(bet) == ExposureDecision::APPROVED);
    REQUIRE(em.player_exposure("p1")  == Approx(30.0).epsilon(1e-6));
    REQUIRE(em.game_exposure("game1") == Approx(30.0).epsilon(1e-6));
    REQUIRE(em.total_in_play()        == Approx(30.0).epsilon(1e-6));
}

TEST_CASE("ExposureManager: bet exactly at player limit is approved", "[exposure]") {
    ExposureManager em(test_config());
    BetRecord bet{"p1", "game1", "WR", 50.0};  // exactly 5% of $1000
    REQUIRE(em.try_place_bet(bet) == ExposureDecision::APPROVED);
}

TEST_CASE("ExposureManager: bet exceeding player limit is rejected", "[exposure]") {
    ExposureManager em(test_config());
    BetRecord bet{"p1", "game1", "WR", 51.0};  // 5.1% — exceeds player cap
    REQUIRE(em.try_place_bet(bet) == ExposureDecision::REJECTED_PLAYER);

    // State must be unchanged after a rejection.
    REQUIRE(em.player_exposure("p1")  == Approx(0.0).epsilon(1e-6));
    REQUIRE(em.total_in_play()        == Approx(0.0).epsilon(1e-6));
}

TEST_CASE("ExposureManager: accumulated player exposure enforced", "[exposure]") {
    ExposureManager em(test_config());
    // First bet: 30 ($30) — approved
    REQUIRE(em.try_place_bet({"p1", "game1", "WR", 30.0}) == ExposureDecision::APPROVED);
    // Second bet: 30+25=55 > 50 max player — rejected
    REQUIRE(em.try_place_bet({"p1", "game1", "WR", 25.0}) == ExposureDecision::REJECTED_PLAYER);
    // Player exposure must still be 30 (second bet not recorded).
    REQUIRE(em.player_exposure("p1") == Approx(30.0).epsilon(1e-6));
}

TEST_CASE("ExposureManager: per-game limit enforced across multiple players", "[exposure]") {
    ExposureManager em(test_config());
    // Place 4 bets of $50 each for different players in the same game.
    // Total game exposure: $200 = 20% of $1000 — at the limit.
    for (int i = 0; i < 4; ++i) {
        BetRecord bet{"p" + std::to_string(i), "game1", "WR", 50.0};
        REQUIRE(em.try_place_bet(bet) == ExposureDecision::APPROVED);
    }
    REQUIRE(em.game_exposure("game1") == Approx(200.0).epsilon(1e-6));

    // 5th bet of $50 would push game total to $250 > $200 — rejected.
    REQUIRE(em.try_place_bet({"p4", "game1", "WR", 50.0}) == ExposureDecision::REJECTED_GAME);
}

TEST_CASE("ExposureManager: total in-play limit enforced across games", "[exposure]") {
    ExposureManager em(test_config());
    // Place 4 bets in 4 different games, $50 each → total = $200 (at the cap).
    for (int i = 0; i < 4; ++i) {
        BetRecord bet{"p" + std::to_string(i),
                      "game" + std::to_string(i), "WR", 50.0};
        REQUIRE(em.try_place_bet(bet) == ExposureDecision::APPROVED);
    }
    REQUIRE(em.total_in_play() == Approx(200.0).epsilon(1e-6));

    // A 5th bet in a new game would push total to $250 > $200 — rejected.
    REQUIRE(em.try_place_bet({"p4", "game4", "WR", 50.0})
            == ExposureDecision::REJECTED_TOTAL);
}

TEST_CASE("ExposureManager: release_bet restores available exposure", "[exposure]") {
    ExposureManager em(test_config());
    BetRecord bet{"p1", "game1", "WR", 50.0};

    REQUIRE(em.try_place_bet(bet) == ExposureDecision::APPROVED);
    em.release_bet(bet);

    REQUIRE(em.player_exposure("p1")  == Approx(0.0).epsilon(1e-6));
    REQUIRE(em.game_exposure("game1") == Approx(0.0).epsilon(1e-6));
    REQUIRE(em.total_in_play()        == Approx(0.0).epsilon(1e-6));

    // After release, the same bet should be approvable again.
    REQUIRE(em.try_place_bet(bet) == ExposureDecision::APPROVED);
}

TEST_CASE("ExposureManager: release_game clears game total and reduces in-play", "[exposure]") {
    ExposureManager em(test_config());
    REQUIRE(em.try_place_bet({"p1", "game1", "WR", 50.0}) == ExposureDecision::APPROVED);
    REQUIRE(em.try_place_bet({"p2", "game1", "WR", 50.0}) == ExposureDecision::APPROVED);
    REQUIRE(em.game_exposure("game1") == Approx(100.0).epsilon(1e-6));

    em.release_game("game1");

    REQUIRE(em.game_exposure("game1") == Approx(0.0).epsilon(1e-6));
    REQUIRE(em.total_in_play()        == Approx(0.0).epsilon(1e-6));
}

TEST_CASE("ExposureManager: release_bet on unknown player is a no-op", "[exposure]") {
    ExposureManager em(test_config());
    // Should not crash or corrupt state.
    em.release_bet({"unknown", "game1", "WR", 50.0});
    REQUIRE(em.total_in_play() == Approx(0.0).epsilon(1e-6));
}

TEST_CASE("ExposureManager: release_game on unknown game is a no-op", "[exposure]") {
    ExposureManager em(test_config());
    em.release_game("nonexistent_game");
    REQUIRE(em.total_in_play() == Approx(0.0).epsilon(1e-6));
}
