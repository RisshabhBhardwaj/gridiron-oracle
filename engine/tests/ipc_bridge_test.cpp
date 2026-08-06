/**
 * engine/tests/ipc_bridge_test.cpp
 *
 * Catch2 v3 unit tests for IpcBridge JSON helpers.
 * Tests parse_event() and serialize_recommendation() without a live socket.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nlohmann/json.hpp>
#include <cstring>

#include "ipc_bridge.hpp"

using namespace gridiron;
using Catch::Approx;

// ── parse_event ──────────────────────────────────────────────────────────────

TEST_CASE("IpcBridge::parse_event: valid JSON round-trip", "[ipc]") {
    const std::string json = R"({
        "player_id":      "00-0032765",
        "game_id":        "2025_01_KC_LAC",
        "event_type":     "projection",
        "stat_delta":     87.5,
        "win_probability": 0.65,
        "decimal_odds":   2.10,
        "timestamp_ms":   1704067200000
    })";

    auto ev_opt = IpcBridge::parse_event(json);
    REQUIRE(ev_opt.has_value());

    const auto& ev = *ev_opt;
    REQUIRE(std::strcmp(ev.player_id,  "00-0032765")      == 0);
    REQUIRE(std::strcmp(ev.game_id,    "2025_01_KC_LAC")  == 0);
    REQUIRE(std::strcmp(ev.event_type, "projection")      == 0);
    REQUIRE(ev.stat_delta      == Approx(87.5f).epsilon(1e-4f));
    REQUIRE(ev.win_probability == Approx(0.65f).epsilon(1e-4f));
    REQUIRE(ev.decimal_odds    == Approx(2.10f).epsilon(1e-4f));
    REQUIRE(ev.timestamp_ms    == 1704067200000LL);
}

TEST_CASE("IpcBridge::parse_event: injury event type parsed correctly", "[ipc]") {
    const std::string json = R"({
        "player_id":      "00-0019596",
        "game_id":        "2025_05_DAL_PHI",
        "event_type":     "injury",
        "stat_delta":     0.0,
        "win_probability": 0.30,
        "decimal_odds":   1.80,
        "timestamp_ms":   1704100000000
    })";

    auto ev_opt = IpcBridge::parse_event(json);
    REQUIRE(ev_opt.has_value());
    REQUIRE(std::strcmp(ev_opt->event_type, "injury") == 0);
    REQUIRE(ev_opt->win_probability == Approx(0.30f).epsilon(1e-4f));
}

TEST_CASE("IpcBridge::parse_event: malformed JSON returns nullopt", "[ipc]") {
    REQUIRE_FALSE(IpcBridge::parse_event("{bad json!!}").has_value());
    REQUIRE_FALSE(IpcBridge::parse_event("").has_value());
    REQUIRE_FALSE(IpcBridge::parse_event("null").has_value());
    REQUIRE_FALSE(IpcBridge::parse_event("[]").has_value());
}

TEST_CASE("IpcBridge::parse_event: missing required field returns nullopt", "[ipc]") {
    // Missing win_probability
    const std::string json = R"({
        "player_id":  "00-0032765",
        "game_id":    "2025_01_KC_LAC",
        "event_type": "projection",
        "stat_delta": 87.5,
        "decimal_odds": 2.10,
        "timestamp_ms": 1704067200000
    })";
    REQUIRE_FALSE(IpcBridge::parse_event(json).has_value());
}

TEST_CASE("IpcBridge::parse_event: player_id truncated to 31 chars + null", "[ipc]") {
    // player_id field in GameEvent is char[32] — long strings must be truncated.
    const std::string long_id(40, 'X');
    const std::string json = R"({"player_id":")" + long_id + R"(","game_id":"g1",
        "event_type":"projection","stat_delta":0,"win_probability":0.5,
        "decimal_odds":2.0,"timestamp_ms":0})";

    auto ev_opt = IpcBridge::parse_event(json);
    REQUIRE(ev_opt.has_value());
    // Stored string must be null-terminated within 32 bytes.
    REQUIRE(std::strlen(ev_opt->player_id) <= 31);
}

// HIGH 4 — buffer overflow: string exactly equal to buffer capacity (sans null)
TEST_CASE("IpcBridge::parse_event: player_id exactly 31 chars is null-terminated", "[ipc][buffer]") {
    // GameEvent::player_id is char[32]. A 31-char input fills bytes [0..30];
    // the explicit null write (buffer[31] = '\0') ensures byte[31] is safe.
    const std::string exact_id(31, 'A');  // 31 chars — fills to index 30
    const std::string json = R"({"player_id":")" + exact_id + R"(","game_id":"g2",
        "event_type":"projection","stat_delta":0,"win_probability":0.5,
        "decimal_odds":2.0,"timestamp_ms":0})";

    auto ev_opt = IpcBridge::parse_event(json);
    REQUIRE(ev_opt.has_value());
    // Must be null-terminated: strlen reads until '\0' without buffer overread.
    REQUIRE(std::strlen(ev_opt->player_id) == 31);
    // The null terminator must be at index 31 (the last byte of the 32-char array).
    REQUIRE(ev_opt->player_id[31] == '\0');
}

TEST_CASE("IpcBridge::parse_event: game_id exactly 31 chars is null-terminated", "[ipc][buffer]") {
    const std::string exact_gid(31, 'G');
    const std::string json = R"({"player_id":"p1","game_id":")" + exact_gid + R"(",
        "event_type":"projection","stat_delta":0,"win_probability":0.5,
        "decimal_odds":2.0,"timestamp_ms":0})";

    auto ev_opt = IpcBridge::parse_event(json);
    REQUIRE(ev_opt.has_value());
    REQUIRE(std::strlen(ev_opt->game_id) == 31);
    REQUIRE(ev_opt->game_id[31] == '\0');
}

TEST_CASE("IpcBridge::parse_event: event_type exactly 15 chars is null-terminated", "[ipc][buffer]") {
    const std::string exact_et(15, 'E');  // event_type is char[16]
    const std::string json = R"({"player_id":"p1","game_id":"g1","event_type":")" + exact_et +
        R"(","stat_delta":0,"win_probability":0.5,"decimal_odds":2.0,"timestamp_ms":0})";

    auto ev_opt = IpcBridge::parse_event(json);
    REQUIRE(ev_opt.has_value());
    REQUIRE(std::strlen(ev_opt->event_type) == 15);
    REQUIRE(ev_opt->event_type[15] == '\0');
}

// HIGH 6 — position field parsed from JSON
TEST_CASE("IpcBridge::parse_event: position field parsed from JSON", "[ipc][position]") {
    const std::string json = R"({
        "player_id":      "00-0032765",
        "game_id":        "2025_01_KC_LAC",
        "event_type":     "projection",
        "position":       "RB",
        "stat_delta":     65.0,
        "win_probability": 0.58,
        "decimal_odds":   2.05,
        "timestamp_ms":   1704067200000
    })";

    auto ev_opt = IpcBridge::parse_event(json);
    REQUIRE(ev_opt.has_value());
    REQUIRE(std::strcmp(ev_opt->position, "RB") == 0);
}

TEST_CASE("IpcBridge::parse_event: position defaults to WR when absent", "[ipc][position]") {
    // Backward compatibility: old payloads without position field → default "WR".
    const std::string json = R"({
        "player_id":      "00-0019596",
        "game_id":        "2025_05_DAL_PHI",
        "event_type":     "projection",
        "stat_delta":     90.0,
        "win_probability": 0.62,
        "decimal_odds":   2.15,
        "timestamp_ms":   1704100000000
    })";

    auto ev_opt = IpcBridge::parse_event(json);
    REQUIRE(ev_opt.has_value());
    REQUIRE(std::strcmp(ev_opt->position, "WR") == 0);
}

TEST_CASE("IpcBridge::parse_event: position exactly 7 chars is null-terminated", "[ipc][buffer][position]") {
    // GameEvent::position is char[8]. A 7-char position string fills bytes [0..6];
    // byte[7] must be '\0'.
    const std::string long_pos(7, 'P');  // 7 chars — fills to index 6
    const std::string json = R"({"player_id":"p1","game_id":"g1","event_type":"projection",
        "position":")" + long_pos + R"(","stat_delta":0,"win_probability":0.5,
        "decimal_odds":2.0,"timestamp_ms":0})";

    auto ev_opt = IpcBridge::parse_event(json);
    REQUIRE(ev_opt.has_value());
    REQUIRE(std::strlen(ev_opt->position) == 7);
    REQUIRE(ev_opt->position[7] == '\0');
}

// ── serialize_recommendation ─────────────────────────────────────────────────

TEST_CASE("IpcBridge::serialize_recommendation: approved bet", "[ipc]") {
    const std::string out = IpcBridge::serialize_recommendation(
        "00-0032765", 0.025, 0.10, true
    );

    auto j = nlohmann::json::parse(out);
    REQUIRE(j["player_id"].get<std::string>()              == "00-0032765");
    REQUIRE(j["recommended_bet_fraction"].get<double>()    == Approx(0.025).epsilon(1e-6));
    REQUIRE(j["kelly_full"].get<double>()                  == Approx(0.10).epsilon(1e-6));
    REQUIRE(j["approved"].get<bool>()                      == true);
}

TEST_CASE("IpcBridge::serialize_recommendation: rejected bet", "[ipc]") {
    const std::string out = IpcBridge::serialize_recommendation(
        "00-0019596", 0.0, 0.0, false
    );

    auto j = nlohmann::json::parse(out);
    REQUIRE(j["approved"].get<bool>()                   == false);
    REQUIRE(j["recommended_bet_fraction"].get<double>() == Approx(0.0).epsilon(1e-9));
}

TEST_CASE("IpcBridge::serialize_recommendation: output is valid JSON", "[ipc]") {
    const std::string out = IpcBridge::serialize_recommendation("p1", 0.03, 0.12, true);
    // nlohmann::json::parse throws on invalid JSON — this verifies it's well-formed.
    REQUIRE_NOTHROW(nlohmann::json::parse(out));
}
