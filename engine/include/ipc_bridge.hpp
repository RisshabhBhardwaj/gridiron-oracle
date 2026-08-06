/**
 * engine/include/ipc_bridge.hpp
 *
 * IpcBridge — Unix domain socket server bridging the Python ML pipeline
 * and the C++ execution engine.
 *
 * Protocol (newline-delimited JSON):
 *
 *   Python → C++ (projection result):
 *     {"player_id":"00-0032765","game_id":"2025_01_KC_LAC",
 *      "event_type":"projection","stat_delta":87.5,
 *      "win_probability":0.65,"decimal_odds":2.10,"timestamp_ms":1704067200000}
 *
 *   C++ → Python (bet sizing response):
 *     {"player_id":"00-0032765","recommended_bet_fraction":0.025,
 *      "kelly_full":0.10,"approved":true}
 *
 * The JSON parse/serialize helpers are static (no instance state) so they
 * can be tested in isolation without a live socket.
 */
#pragma once

#include "game_event.hpp"
#include "ring_buffer.hpp"
#include <optional>
#include <string>

namespace gridiron {

class IpcBridge {
public:
    using Buffer = RingBuffer<GameEvent, 1024>;

    explicit IpcBridge(Buffer& ring_buf,
                       std::string socket_path = "/tmp/gridiron.sock")
        : ring_buf_(ring_buf), socket_path_(std::move(socket_path)) {}

    ~IpcBridge();

    /** Open the Unix socket and begin accepting connections. */
    void start();

    /** Close the socket and clean up the socket file. */
    void stop();

    // ── Static JSON helpers (public for unit testing) ──────────────────────

    /**
     * parse_event — Deserialize a JSON string into a GameEvent.
     * Returns nullopt if the JSON is malformed or any required field is missing.
     */
    static std::optional<GameEvent> parse_event(const std::string& json_str);

    /**
     * serialize_recommendation — Serialize a bet sizing result to a JSON string.
     * Returned string is suitable for sending back to Python over the socket.
     */
    static std::string serialize_recommendation(const std::string& player_id,
                                                double bet_fraction,
                                                double kelly_full,
                                                bool   approved);

private:
    Buffer&     ring_buf_;
    std::string socket_path_;
    int         server_fd_ = -1;
    bool        running_   = false;
};

}  // namespace gridiron
