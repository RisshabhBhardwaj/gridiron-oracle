/**
 * engine/src/ipc_bridge.cpp
 *
 * Unix domain socket server implementation.
 * JSON parsing/serialization via nlohmann/json.
 */
#include "ipc_bridge.hpp"

#include <nlohmann/json.hpp>
#include <cstring>
#include <stdexcept>

// POSIX socket headers (macOS / Linux)
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace gridiron {

IpcBridge::~IpcBridge() {
    stop();
}

void IpcBridge::start() {
    // ring_buf_ is used in the Phase 4 accept-loop thread; suppress the
    // unused-private-field warning until that loop is implemented.
    (void)ring_buf_;

    server_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        throw std::runtime_error("IpcBridge: failed to create Unix socket");
    }

    // Remove any stale socket file from a previous run.
    ::unlink(socket_path_.c_str());

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(server_fd_,
               reinterpret_cast<sockaddr*>(&addr),
               static_cast<socklen_t>(sizeof(addr))) < 0) {
        ::close(server_fd_);
        server_fd_ = -1;
        throw std::runtime_error("IpcBridge: bind failed — " + socket_path_);
    }

    if (::listen(server_fd_, 4) < 0) {
        ::close(server_fd_);
        server_fd_ = -1;
        throw std::runtime_error("IpcBridge: listen failed");
    }

    running_ = true;
    // Full accept-loop thread is Phase 4 — launched here when WebSocket feed
    // is live. For Phase 3, ring buffer is populated via parse_event() in tests.
}

void IpcBridge::stop() {
    running_ = false;
    if (server_fd_ >= 0) {
        ::close(server_fd_);
        server_fd_ = -1;
        ::unlink(socket_path_.c_str());
    }
}

std::optional<GameEvent> IpcBridge::parse_event(const std::string& json_str) {
    try {
        auto j = nlohmann::json::parse(json_str);

        GameEvent ev{};
        std::strncpy(ev.player_id,
                     j.at("player_id").get<std::string>().c_str(),
                     sizeof(ev.player_id) - 1);
        ev.player_id[sizeof(ev.player_id) - 1] = '\0';   // guarantee null termination

        std::strncpy(ev.game_id,
                     j.at("game_id").get<std::string>().c_str(),
                     sizeof(ev.game_id) - 1);
        ev.game_id[sizeof(ev.game_id) - 1] = '\0';       // guarantee null termination

        std::strncpy(ev.event_type,
                     j.at("event_type").get<std::string>().c_str(),
                     sizeof(ev.event_type) - 1);
        ev.event_type[sizeof(ev.event_type) - 1] = '\0'; // guarantee null termination

        // position is optional for backward compatibility; defaults to "WR".
        const std::string pos = j.value("position", "WR");
        std::strncpy(ev.position, pos.c_str(), sizeof(ev.position) - 1);
        ev.position[sizeof(ev.position) - 1] = '\0';     // guarantee null termination

        ev.stat_delta      = j.at("stat_delta").get<float>();
        ev.win_probability = j.at("win_probability").get<float>();
        ev.decimal_odds    = j.at("decimal_odds").get<float>();
        ev.timestamp_ms    = j.at("timestamp_ms").get<int64_t>();

        return ev;

    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    }
}

std::string IpcBridge::serialize_recommendation(const std::string& player_id,
                                                 double bet_fraction,
                                                 double kelly_full,
                                                 bool   approved) {
    nlohmann::json j{
        {"player_id",                player_id},
        {"recommended_bet_fraction", bet_fraction},
        {"kelly_full",               kelly_full},
        {"approved",                 approved},
    };
    return j.dump();
}

}  // namespace gridiron
