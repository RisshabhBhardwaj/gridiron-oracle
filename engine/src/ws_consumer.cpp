/**
 * engine/src/ws_consumer.cpp
 *
 * WebSocket consumer — reads live GameEvent JSON from a WebSocket endpoint
 * (e.g. the FastAPI /alerts/ws endpoint) and pushes deserialized GameEvent
 * structs into the SPSC ring buffer for consumption by SignalProcessor.
 *
 * Build requirements:
 *   macOS:  brew install boost
 *   Ubuntu: sudo apt-get install libboost-all-dev
 *
 * The HAVE_BOOST preprocessor guard (set by CMakeLists.txt when Boost is found)
 * allows the engine to compile and link cleanly without Boost installed.
 * Without Boost, run() and stop() are empty stubs.
 */

#ifdef HAVE_BOOST

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>

namespace beast     = boost::beast;
namespace websocket = beast::websocket;
namespace net       = boost::asio;
using tcp           = net::ip::tcp;

#endif  // HAVE_BOOST

#include "ws_consumer.hpp"
#include "ipc_bridge.hpp"

#include <chrono>
#include <iostream>
#include <thread>

namespace gridiron {

WebSocketConsumer::WebSocketConsumer(RingBuffer<GameEvent, 1024>& buf,
                                     std::string host,
                                     std::string port,
                                     std::string path)
    : buf_(buf)
    , host_(std::move(host))
    , port_(std::move(port))
    , path_(std::move(path))
{}

#ifdef HAVE_BOOST

bool WebSocketConsumer::_do_session() {
    try {
        net::io_context ioc;
        tcp::resolver resolver{ioc};
        websocket::stream<tcp::socket> ws{ioc};

        // Resolve and connect TCP layer.
        auto const results = resolver.resolve(host_, port_);
        net::connect(ws.next_layer(), results);

        // Set HTTP upgrade headers before handshake.
        ws.set_option(websocket::stream_base::decorator(
            [this](websocket::request_type& req) {
                req.set(beast::http::field::user_agent, "gridiron-engine/1.0");
                req.set(beast::http::field::host, host_ + ":" + port_);
            }));

        // Perform the WebSocket handshake.
        ws.handshake(host_ + ":" + port_, path_);

        // Read loop — one Beast read call per message (synchronous).
        while (running_.load(std::memory_order_acquire)) {
            beast::flat_buffer buffer;
            boost::system::error_code ec;
            ws.read(buffer, ec);

            if (ec == websocket::error::closed) {
                // Server closed cleanly — trigger backoff / reconnect.
                return false;
            }
            if (ec) {
                std::cerr << "[ws_consumer] read error: " << ec.message() << '\n';
                return false;
            }

            // Deserialize JSON payload → GameEvent.
            std::string payload = beast::buffers_to_string(buffer.data());
            auto ev = IpcBridge::parse_event(payload);
            if (ev) {
                // Spin up to 10 µs if the ring buffer is momentarily full.
                // Drop the event rather than block indefinitely.
                auto deadline = std::chrono::steady_clock::now()
                              + std::chrono::microseconds(10);
                while (!buf_.push(*ev)) {
                    if (std::chrono::steady_clock::now() >= deadline) {
                        break;
                    }
                    std::this_thread::yield();
                }
            }
        }

        // running_ was cleared by stop() — close gracefully.
        try {
            ws.close(websocket::close_code::normal);
        } catch (...) {}
        return true;  // clean shutdown — do not reconnect

    } catch (const std::exception& e) {
        std::cerr << "[ws_consumer] session exception: " << e.what() << '\n';
        return false;  // error — trigger backoff and reconnect
    }
}

void WebSocketConsumer::run() {
    int backoff_s     = 1;
    const int max_bk  = 30;

    while (running_.load(std::memory_order_acquire)) {
        bool clean = _do_session();
        if (clean) break;  // stop() was called — no reconnect

        // Exponential backoff: poll every 100 ms so stop() is responsive.
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::seconds(backoff_s);
        while (running_.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        backoff_s = std::min(backoff_s * 2, max_bk);
    }
}

void WebSocketConsumer::stop() {
    // Release store — happens-before any subsequent acquire load in run()/_do_session().
    running_.store(false, std::memory_order_release);
}

#else  // HAVE_BOOST not defined — empty stubs

bool WebSocketConsumer::_do_session() { return true; }

void WebSocketConsumer::run() {}

void WebSocketConsumer::stop() {
    running_.store(false, std::memory_order_release);
}

#endif  // HAVE_BOOST

}  // namespace gridiron
