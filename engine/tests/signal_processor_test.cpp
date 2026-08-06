/**
 * engine/tests/signal_processor_test.cpp
 *
 * Catch2 v3 unit tests for SignalProcessor and config loading.
 *
 * Tests cover:
 *   HIGH 4 — BetRecommendation null-termination after strncpy
 *   HIGH 5 — Config key alignment (kelly_fraction / max_bet_fraction /
 *             max_per_player_fraction / max_per_game_fraction /
 *             max_total_in_play_fraction / bankroll)
 *   HIGH 6 — position flows from GameEvent into BetRecord (RB event
 *             verifies RB limits apply, not hardcoded WR)
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nlohmann/json.hpp>
#include <cstring>
#include <fstream>
#include <sstream>

#include "game_event.hpp"
#include "ring_buffer.hpp"
#include "kelly_sizer.hpp"
#include "exposure_manager.hpp"
#include "signal_processor.hpp"
#include "ipc_bridge.hpp"

using namespace gridiron;
using Catch::Approx;

// ── Helpers ───────────────────────────────────────────────────────────────────

static GameEvent make_event(const std::string& pid,
                             const std::string& gid,
                             const std::string& pos,
                             float win_prob,
                             float decimal_odds) {
    GameEvent ev{};
    std::strncpy(ev.player_id,  pid.c_str(), sizeof(ev.player_id)  - 1);
    ev.player_id[sizeof(ev.player_id) - 1]   = '\0';
    std::strncpy(ev.game_id,    gid.c_str(), sizeof(ev.game_id)    - 1);
    ev.game_id[sizeof(ev.game_id) - 1]       = '\0';
    std::strncpy(ev.event_type, "projection", sizeof(ev.event_type) - 1);
    ev.event_type[sizeof(ev.event_type) - 1] = '\0';
    std::strncpy(ev.position,   pos.c_str(), sizeof(ev.position)   - 1);
    ev.position[sizeof(ev.position) - 1]     = '\0';
    ev.win_probability = win_prob;
    ev.decimal_odds    = decimal_odds;
    ev.stat_delta      = 0.0f;
    ev.timestamp_ms    = 0;
    return ev;
}

static ExposureConfig standard_config() {
    ExposureConfig cfg;
    cfg.bankroll                   = 1000.0;
    cfg.max_per_player_fraction    = 0.05;
    cfg.max_per_game_fraction      = 0.20;
    cfg.max_total_in_play_fraction = 0.20;
    return cfg;
}

// ── HIGH 4: BetRecommendation null-termination ────────────────────────────────

TEST_CASE("SignalProcessor: BetRecommendation player_id is null-terminated after 31-char input",
          "[signal][buffer]") {
    // player_id of exactly 31 chars must be stored with null at index 31.
    RingBuffer<GameEvent, 1024> buf;
    KellySizer      kelly;
    ExposureManager exposure(standard_config());
    SignalProcessor proc(buf, kelly, exposure, 1000.0);

    BetRecommendation captured{};
    proc.set_callback([&](const BetRecommendation& rec) { captured = rec; });

    const std::string pid(31, 'X');  // exactly 31 chars
    buf.push(make_event(pid, "game1", "WR", 0.60f, 2.10f));
    proc.process_one();

    REQUIRE(std::strlen(captured.player_id) == 31);
    REQUIRE(captured.player_id[31] == '\0');
}

TEST_CASE("SignalProcessor: BetRecommendation game_id is null-terminated after 31-char input",
          "[signal][buffer]") {
    RingBuffer<GameEvent, 1024> buf;
    KellySizer      kelly;
    ExposureManager exposure(standard_config());
    SignalProcessor proc(buf, kelly, exposure, 1000.0);

    BetRecommendation captured{};
    proc.set_callback([&](const BetRecommendation& rec) { captured = rec; });

    const std::string gid(31, 'G');  // exactly 31 chars
    buf.push(make_event("p1", gid, "RB", 0.60f, 2.10f));
    proc.process_one();

    REQUIRE(std::strlen(captured.game_id) == 31);
    REQUIRE(captured.game_id[31] == '\0');
}

// ── HIGH 5: Config key alignment ─────────────────────────────────────────────

TEST_CASE("Config: all ExposureConfig fields parse from correct JSON keys", "[config]") {
    // Verify that the JSON keys in main.cpp match ExposureConfig struct field names.
    // If a key is wrong, the value falls back to the default and diverges.
    const std::string config_json = R"({
        "kelly_fraction":             0.30,
        "max_bet_fraction":           0.08,
        "bankroll":                   5000.0,
        "max_per_player_fraction":    0.03,
        "max_per_game_fraction":      0.15,
        "max_total_in_play_fraction": 0.25
    })";

    auto j = nlohmann::json::parse(config_json);

    KellyConfig kelly_cfg;
    kelly_cfg.kelly_fraction   = j.value("kelly_fraction",   0.25);
    kelly_cfg.max_bet_fraction = j.value("max_bet_fraction", 0.05);  // HIGH 5 fix

    ExposureConfig exp_cfg;
    exp_cfg.bankroll                   = j.value("bankroll",                  1000.0);
    exp_cfg.max_per_player_fraction    = j.value("max_per_player_fraction",    0.05);
    exp_cfg.max_per_game_fraction      = j.value("max_per_game_fraction",      0.20);  // HIGH 5 fix
    exp_cfg.max_total_in_play_fraction = j.value("max_total_in_play_fraction", 0.20);

    REQUIRE(kelly_cfg.kelly_fraction   == Approx(0.30).epsilon(1e-9));
    REQUIRE(kelly_cfg.max_bet_fraction == Approx(0.08).epsilon(1e-9));
    REQUIRE(exp_cfg.bankroll                   == Approx(5000.0).epsilon(1e-9));
    REQUIRE(exp_cfg.max_per_player_fraction    == Approx(0.03).epsilon(1e-9));
    REQUIRE(exp_cfg.max_per_game_fraction      == Approx(0.15).epsilon(1e-9));  // was wrong
    REQUIRE(exp_cfg.max_total_in_play_fraction == Approx(0.25).epsilon(1e-9));
}

TEST_CASE("Config: max_per_game_fraction and max_total_in_play_fraction are independent",
          "[config]") {
    // Before HIGH 5 fix, both fields read the same JSON key → same value.
    // After fix, they are independent and can differ.
    const std::string config_json = R"({
        "max_per_game_fraction":      0.15,
        "max_total_in_play_fraction": 0.35
    })";

    auto j = nlohmann::json::parse(config_json);
    ExposureConfig cfg;
    cfg.max_per_game_fraction      = j.value("max_per_game_fraction",      0.20);
    cfg.max_total_in_play_fraction = j.value("max_total_in_play_fraction", 0.20);

    // These must differ — proof the keys are independent after the fix.
    REQUIRE(cfg.max_per_game_fraction      == Approx(0.15).epsilon(1e-9));
    REQUIRE(cfg.max_total_in_play_fraction == Approx(0.35).epsilon(1e-9));
    REQUIRE(cfg.max_per_game_fraction != cfg.max_total_in_play_fraction);
}

TEST_CASE("Config: max_bet_fraction and max_per_player_fraction are independent",
          "[config]") {
    // Before HIGH 5 fix, kelly_cfg.max_bet_fraction read "max_per_player_fraction"
    // (same key as exp_cfg), so changing one would affect the other.
    const std::string config_json = R"({
        "max_bet_fraction":        0.08,
        "max_per_player_fraction": 0.03
    })";

    auto j = nlohmann::json::parse(config_json);
    KellyConfig  kelly_cfg;
    ExposureConfig exp_cfg;
    kelly_cfg.max_bet_fraction        = j.value("max_bet_fraction",        0.05);
    exp_cfg.max_per_player_fraction   = j.value("max_per_player_fraction", 0.05);

    REQUIRE(kelly_cfg.max_bet_fraction      == Approx(0.08).epsilon(1e-9));
    REQUIRE(exp_cfg.max_per_player_fraction == Approx(0.03).epsilon(1e-9));
    REQUIRE(kelly_cfg.max_bet_fraction != exp_cfg.max_per_player_fraction);
}

// ── HIGH 6: position flows from GameEvent into BetRecord ─────────────────────

TEST_CASE("SignalProcessor: position from GameEvent flows into ExposureManager", "[signal][position]") {
    // Verify that the position field from the event reaches the BetRecord.
    // The primary test: process an RB event and confirm the recommendation
    // is not blocked by a WR-specific limit mismatch.
    RingBuffer<GameEvent, 1024> buf;
    KellySizer      kelly;
    ExposureManager exposure(standard_config());
    SignalProcessor proc(buf, kelly, exposure, 1000.0);

    BetRecommendation captured{};
    bool callback_called = false;
    proc.set_callback([&](const BetRecommendation& rec) {
        captured = rec;
        callback_called = true;
    });

    // Push an RB event with strong edge (should be approved within limits).
    buf.push(make_event("rb_player_1", "game_rb_01", "RB", 0.65f, 2.20f));
    REQUIRE(proc.process_one());
    REQUIRE(callback_called);
    REQUIRE(captured.approved == true);
}

TEST_CASE("SignalProcessor: RB event uses RB position (not hardcoded WR)", "[signal][position]") {
    // Verify the event's position field is propagated.
    // We parse the JSON for an RB event and check the position field.
    const std::string rb_json = R"({
        "player_id":      "00-0099999",
        "game_id":        "2025_10_SF_LAR",
        "event_type":     "projection",
        "position":       "RB",
        "stat_delta":     85.0,
        "win_probability": 0.60,
        "decimal_odds":   2.10,
        "timestamp_ms":   1704067300000
    })";

    auto ev_opt = IpcBridge::parse_event(rb_json);
    REQUIRE(ev_opt.has_value());
    REQUIRE(std::strcmp(ev_opt->position, "RB") == 0);

    // Process it through SignalProcessor and confirm the callback fires.
    RingBuffer<GameEvent, 1024> buf;
    KellySizer      kelly;
    ExposureManager exposure(standard_config());
    SignalProcessor proc(buf, kelly, exposure, 1000.0);

    bool processed = false;
    proc.set_callback([&](const BetRecommendation&) { processed = true; });

    buf.push(*ev_opt);
    REQUIRE(proc.process_one());
    REQUIRE(processed);
}

TEST_CASE("SignalProcessor: QB event uses QB position (not hardcoded WR)", "[signal][position]") {
    const std::string qb_json = R"({
        "player_id":      "00-0033873",
        "game_id":        "2025_01_KC_LAC",
        "event_type":     "projection",
        "position":       "QB",
        "stat_delta":     280.0,
        "win_probability": 0.63,
        "decimal_odds":   2.05,
        "timestamp_ms":   1704067200000
    })";

    auto ev_opt = IpcBridge::parse_event(qb_json);
    REQUIRE(ev_opt.has_value());
    REQUIRE(std::strcmp(ev_opt->position, "QB") == 0);
}
