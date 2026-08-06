/**
 * engine/src/main.cpp
 *
 * Gridiron Engine entry point.
 *
 * Wires together:
 *   IpcBridge → RingBuffer<GameEvent, 1024> → SignalProcessor
 *                                                ↓
 *                                         KellySizer → ExposureManager
 *
 * Config loaded from engine_config.json (path configurable via argv[1]).
 * Handles SIGINT / SIGTERM for clean shutdown.
 */
#include "ring_buffer.hpp"
#include "game_event.hpp"
#include "kelly_sizer.hpp"
#include "exposure_manager.hpp"
#include "signal_processor.hpp"
#include "ipc_bridge.hpp"
#ifdef HAVE_BOOST
#include "ws_consumer.hpp"
#endif

#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>

namespace {
std::atomic<bool> g_running{true};
void sig_handler(int) noexcept { g_running.store(false, std::memory_order_relaxed); }
}

int main(int argc, char* argv[]) {
    const std::string config_path = (argc > 1) ? argv[1] : "config/engine_config.json";

    gridiron::KellyConfig    kelly_cfg;
    gridiron::ExposureConfig exp_cfg;

    // Load config — non-fatal if missing (use defaults).
    try {
        std::ifstream f(config_path);
        if (f.is_open()) {
            auto j = nlohmann::json::parse(f);
            kelly_cfg.kelly_fraction     = j.value("kelly_fraction",             0.25);
            kelly_cfg.max_bet_fraction   = j.value("max_bet_fraction",           0.05);
            exp_cfg.bankroll                   = j.value("bankroll",                  1000.0);
            exp_cfg.max_per_player_fraction    = j.value("max_per_player_fraction",    0.05);
            exp_cfg.max_per_game_fraction      = j.value("max_per_game_fraction",      0.20);
            exp_cfg.max_total_in_play_fraction = j.value("max_total_in_play_fraction", 0.20);
        }
    } catch (const std::exception& e) {
        std::cerr << "[engine] Config load failed (" << e.what() << ") — using defaults\n";
    }

    // Build the pipeline.
    gridiron::RingBuffer<gridiron::GameEvent, 1024> ring_buf;
    gridiron::KellySizer      kelly(kelly_cfg);
    gridiron::ExposureManager exposure(exp_cfg);
    gridiron::SignalProcessor processor(ring_buf, kelly, exposure, exp_cfg.bankroll);

    processor.set_callback([](const gridiron::BetRecommendation& rec) {
        std::cout << "[engine] player=" << rec.player_id
                  << " approved=" << (rec.approved ? "YES" : "NO")
                  << " fraction=" << rec.recommended_bet_fraction
                  << " kelly_full=" << rec.kelly_full << '\n';
    });

    gridiron::IpcBridge bridge(ring_buf);

    std::signal(SIGINT,  sig_handler);
    std::signal(SIGTERM, sig_handler);

    try {
        bridge.start();
        std::cout << "[engine] Listening on /tmp/gridiron.sock\n";
    } catch (const std::exception& e) {
        std::cerr << "[engine] Failed to start IPC bridge: " << e.what() << '\n';
        return 1;
    }

#ifdef HAVE_BOOST
    // WebSocket consumer — feeds live GameEvents from FastAPI /alerts/ws into
    // the ring buffer.  Runs in its own thread; reconnects automatically.
    gridiron::WebSocketConsumer ws_consumer(ring_buf, "localhost", "8000", "/alerts/ws");
    std::thread ws_thread([&ws_consumer]{ ws_consumer.run(); });
#endif

    // Consumer loop — sleep-yields when buffer is empty.
    while (g_running.load(std::memory_order_relaxed)) {
        if (!processor.process_one()) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

#ifdef HAVE_BOOST
    ws_consumer.stop();
    ws_thread.join();
#endif

    bridge.stop();
    std::cout << "[engine] Shutdown complete.\n";
    return 0;
}
