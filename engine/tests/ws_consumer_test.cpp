/**
 * engine/tests/ws_consumer_test.cpp
 *
 * Integration test for WebSocketConsumer.
 *
 * Requires: HAVE_BOOST (compiled with ws_integration_tests target).
 *
 * Test design:
 *   1. Spin up a sync Beast WebSocket acceptor on 127.0.0.1:18765.
 *   2. Signal ready_flag when listening.
 *   3. Accept one connection, send 3 JSON GameEvent messages, then close.
 *   4. WebSocketConsumer connects, reads events, pushes to ring buffer.
 *   5. stop() is called after 500 ms (gives events time to arrive).
 *   6. Assert: 3 events in buffer with correct player_id / position;
 *              4th pop() returns false (buffer empty).
 */

#ifdef HAVE_BOOST

#include <catch2/catch_test_macros.hpp>

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include "ws_consumer.hpp"
#include "ring_buffer.hpp"
#include "game_event.hpp"

namespace beast     = boost::beast;
namespace websocket = beast::websocket;
namespace net       = boost::asio;
using tcp           = net::ip::tcp;

using namespace gridiron;

// ── Mock server constants ─────────────────────────────────────────────────────

static constexpr const char* MOCK_HOST = "127.0.0.1";

// 3 valid GameEvent JSON payloads — fields match IpcBridge::parse_event() schema.
static const std::string EVENT_JSONS[3] = {
    R"({"player_id":"P001","game_id":"G001","event_type":"projection","position":"WR","stat_delta":75.0,"win_probability":0.60,"decimal_odds":1.85,"timestamp_ms":1700000001000})",
    R"({"player_id":"P002","game_id":"G001","event_type":"projection","position":"RB","stat_delta":65.0,"win_probability":0.55,"decimal_odds":2.10,"timestamp_ms":1700000002000})",
    R"({"player_id":"P003","game_id":"G002","event_type":"projection","position":"TE","stat_delta":45.0,"win_probability":0.58,"decimal_odds":1.95,"timestamp_ms":1700000003000})",
};

// ── Mock server ───────────────────────────────────────────────────────────────

/**
 * run_mock_server
 *
 * Starts a synchronous Beast WebSocket acceptor on 127.0.0.1:18765.
 * Sets ready_flag = true when the acceptor is bound and listening.
 * Accepts exactly one connection, sends 3 JSON events, then closes.
 *
 * Runs in a background std::thread during the test.
 */
struct MockServerState {
    std::condition_variable cv;
    std::mutex mutex;
    bool ready = false;
    unsigned short port = 0;
    std::string error;
};

static void run_mock_server(MockServerState& state) {
    try {
        net::io_context ioc;
        tcp::acceptor acceptor{ioc};
        acceptor.open(tcp::v4());
        acceptor.set_option(net::socket_base::reuse_address(true));
        acceptor.bind(tcp::endpoint{net::ip::make_address(MOCK_HOST), 0});
        acceptor.listen();

        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.port = acceptor.local_endpoint().port();
            state.ready = true;
        }
        state.cv.notify_one();

        tcp::socket raw_socket{ioc};
        acceptor.accept(raw_socket);

        websocket::stream<tcp::socket> ws{std::move(raw_socket)};
        ws.accept();  // HTTP → WebSocket upgrade

        for (const auto& payload : EVENT_JSONS) {
            ws.write(net::buffer(payload));
        }

        ws.close(websocket::close_code::normal);

    } catch (const std::exception& e) {
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.ready = true;
            state.error = e.what();
        }
        state.cv.notify_one();
    }
}

// ── Integration test case ─────────────────────────────────────────────────────

TEST_CASE("WebSocketConsumer: receives 3 events from mock server", "[ws][integration]") {
    RingBuffer<GameEvent, 1024> buf;
    MockServerState server_state;

    // Start mock server in a background thread.
    std::thread server_thread(run_mock_server, std::ref(server_state));

    // Wait up to 2 seconds for the server to start listening.
    {
        std::unique_lock<std::mutex> lock(server_state.mutex);
        const bool ready = server_state.cv.wait_for(
            lock,
            std::chrono::seconds(2),
            [&server_state] { return server_state.ready; }
        );
        if (!ready) {
            lock.unlock();
            server_thread.join();
            FAIL("Mock server did not start within 2 seconds");
        }
        if (!server_state.error.empty()) {
            const std::string error = server_state.error;
            lock.unlock();
            server_thread.join();
            FAIL(error);
        }
    }

    WebSocketConsumer consumer(buf, MOCK_HOST, std::to_string(server_state.port));

    // Stop consumer after 500 ms — enough time to receive all 3 events and
    // process the server-initiated close.
    std::thread stop_thread([&consumer] {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        consumer.stop();
    });

    // run() blocks until stop() is called (or server disconnects + backoff expires).
    consumer.run();

    stop_thread.join();
    server_thread.join();

    // ── Assertions ────────────────────────────────────────────────────────────

    GameEvent ev;

    REQUIRE(buf.pop(ev));
    REQUIRE(std::string(ev.player_id) == "P001");
    REQUIRE(std::string(ev.position)  == "WR");

    REQUIRE(buf.pop(ev));
    REQUIRE(std::string(ev.player_id) == "P002");
    REQUIRE(std::string(ev.position)  == "RB");

    REQUIRE(buf.pop(ev));
    REQUIRE(std::string(ev.player_id) == "P003");
    REQUIRE(std::string(ev.position)  == "TE");

    // Buffer must be empty after exactly 3 events.
    REQUIRE_FALSE(buf.pop(ev));
}

#endif  // HAVE_BOOST
