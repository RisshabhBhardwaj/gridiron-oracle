#pragma once
#include "ring_buffer.hpp"
#include "game_event.hpp"
#include <atomic>
#include <string>

namespace gridiron {

/**
 * WebSocketConsumer — reads live GameEvent JSON from a WebSocket endpoint
 * and pushes events into the SPSC ring buffer.
 *
 * Build requirement: Boost 1.82+ (Boost.Beast + ASIO).
 *   macOS:  brew install boost
 *   Ubuntu: sudo apt-get install libboost-all-dev
 *
 * Without Boost (HAVE_BOOST not defined), run() and stop() are no-ops so
 * the engine binary still compiles and links cleanly.
 */
class WebSocketConsumer {
public:
    WebSocketConsumer(RingBuffer<GameEvent, 1024>& buf,
                      std::string host,
                      std::string port,
                      std::string path = "/ws/live-events");

    // Blocks until stop() is called. Reconnects with exponential backoff (max 30s).
    void run();

    // Thread-safe: atomic release store → sets running_ = false.
    // After stop(), run() drains cleanly and returns.
    void stop();

private:
    RingBuffer<GameEvent, 1024>& buf_;
    std::string host_;
    std::string port_;
    std::string path_;

    // Memory ordering: release on write (stop()), acquire on reads (run()/_do_session()).
    std::atomic<bool> running_{true};

    // Manages one connection lifetime.
    // Returns true  = clean shutdown (stop() was called).
    // Returns false = error or server disconnect (triggers backoff + reconnect).
    bool _do_session();
};

}  // namespace gridiron
