/**
 * engine/include/game_event.hpp
 *
 * GameEvent — the fundamental message unit passed through the SPSC ring buffer.
 *
 * Design constraints (CLAUDE.md §3 C++ Rules):
 *   - No dynamic allocation: all fields are fixed-size arrays or POD types.
 *   - No std::string in the hot path.
 *   - Stack-allocatable and trivially copy-constructible into the ring buffer.
 */
#pragma once

#include <cstdint>

namespace gridiron {

struct GameEvent {
    char    player_id[32];      // gsis_id e.g. "00-0032765"
    char    game_id[32];        // e.g. "2025_01_KC_LAC"
    char    event_type[16];     // "projection" | "score" | "injury"
    char    position[8];        // "QB" | "RB" | "WR" | "TE" — used for exposure limits
    float   stat_delta;         // change in stat value (yards, receptions, etc.)
    float   win_probability;    // boom_probability from ML pipeline [0, 1]
    float   decimal_odds;       // market decimal odds e.g. 2.10 = $1.10 profit per $1
    int64_t timestamp_ms;       // Unix epoch in milliseconds
};

static_assert(sizeof(GameEvent) <= 256,
              "GameEvent must fit in 256 bytes to avoid cache thrashing");

}  // namespace gridiron
