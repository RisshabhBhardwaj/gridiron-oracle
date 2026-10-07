/**
 * engine/tests/ring_buffer_test.cpp
 *
 * Catch2 v3 correctness tests + BENCHMARK for RingBuffer<GameEvent, N>.
 * BENCHMARK target: < 200 ns per push+pop round-trip.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <thread>
#include <vector>
#include <cstring>

#include "ring_buffer.hpp"
#include "game_event.hpp"

using namespace gridiron;
using Catch::Approx;

// ── Test fixture helper ──────────────────────────────────────────────────────

static GameEvent make_event(const char* player_id, float prob = 0.60f) {
    GameEvent ev{};
    std::strncpy(ev.player_id,  player_id,        sizeof(ev.player_id)  - 1);
    std::strncpy(ev.game_id,    "2025_01_KC_LAC", sizeof(ev.game_id)    - 1);
    std::strncpy(ev.event_type, "projection",     sizeof(ev.event_type) - 1);
    ev.stat_delta      = 87.5f;
    ev.win_probability = prob;
    ev.decimal_odds    = 2.10f;
    ev.timestamp_ms    = 1704067200000LL;
    return ev;
}

// ── Correctness tests ────────────────────────────────────────────────────────

TEST_CASE("RingBuffer: basic push/pop round-trip", "[ring_buffer]") {
    RingBuffer<GameEvent, 4> buf;
    auto ev = make_event("00-0032765");

    REQUIRE(buf.push(ev));

    GameEvent out{};
    REQUIRE(buf.pop(out));
    REQUIRE(std::strcmp(out.player_id, "00-0032765") == 0);
    REQUIRE(out.win_probability == Approx(0.60f).epsilon(1e-5f));
}

TEST_CASE("RingBuffer: empty buffer pop returns false", "[ring_buffer]") {
    RingBuffer<GameEvent, 4> buf;
    GameEvent out{};
    REQUIRE_FALSE(buf.pop(out));
    REQUIRE(buf.empty());
}

TEST_CASE("RingBuffer: full buffer push returns false", "[ring_buffer]") {
    // Capacity=4 can hold 3 items (one slot is always the sentinel gap).
    RingBuffer<GameEvent, 4> buf;
    auto ev = make_event("p1");
    REQUIRE(buf.push(ev));
    REQUIRE(buf.push(ev));
    REQUIRE(buf.push(ev));
    REQUIRE_FALSE(buf.push(ev));   // 4th push fails — full
}

TEST_CASE("RingBuffer: FIFO order preserved across multiple items", "[ring_buffer]") {
    RingBuffer<GameEvent, 8> buf;
    for (int i = 0; i < 5; ++i) {
        auto ev = make_event("p1", 0.50f + 0.01f * static_cast<float>(i));
        REQUIRE(buf.push(ev));
    }
    for (int i = 0; i < 5; ++i) {
        GameEvent out{};
        REQUIRE(buf.pop(out));
        REQUIRE(out.win_probability == Approx(0.50f + 0.01f * static_cast<float>(i)).epsilon(1e-5f));
    }
}

TEST_CASE("RingBuffer: wraparound correctness at power-of-2 boundary", "[ring_buffer]") {
    // Capacity=4 → max 3 items. Repeated push/pop exercises the index wraparound.
    RingBuffer<GameEvent, 4> buf;
    auto ev = make_event("wrap");

    for (int round = 0; round < 8; ++round) {
        REQUIRE(buf.push(ev));
        REQUIRE(buf.push(ev));
        REQUIRE(buf.push(ev));
        REQUIRE_FALSE(buf.push(ev));  // full

        GameEvent out{};
        REQUIRE(buf.pop(out));
        REQUIRE(buf.pop(out));
        REQUIRE(buf.pop(out));
        REQUIRE_FALSE(buf.pop(out)); // empty
    }
}

TEST_CASE("RingBuffer: cache-line alignment verified", "[ring_buffer]") {
    using RB = RingBuffer<GameEvent, 1024>;
    using PA = RB::PaddedAtomic;
    REQUIRE(sizeof(PA) == RB::kCacheLine);
    REQUIRE(alignof(PA) == RB::kCacheLine);
}

TEST_CASE("RingBuffer: producer-consumer thread safety (10k round-trips)", "[ring_buffer]") {
    RingBuffer<GameEvent, 1024> buf;
    constexpr int N = 10'000;
    std::vector<int> received;
    received.reserve(N);

    // Producer: sends events with win_probability encoding the sequence index.
    std::thread producer([&]() {
        for (int i = 0; i < N; ++i) {
            auto ev = make_event("thread_p", static_cast<float>(i) / static_cast<float>(N));
            while (!buf.push(ev)) { /* spin on full buffer */ }
        }
    });

    // Consumer: decodes sequence index back from win_probability.
    std::thread consumer([&]() {
        for (int i = 0; i < N; ++i) {
            GameEvent out{};
            while (!buf.pop(out)) { /* spin on empty buffer */ }
            // Recover integer index: round(prob * N)
            received.push_back(static_cast<int>(out.win_probability * static_cast<float>(N) + 0.5f));
        }
    });

    producer.join();
    consumer.join();

    REQUIRE(static_cast<int>(received.size()) == N);
    // FIFO: indices must arrive in order 0, 1, ..., N-1.
    for (int i = 0; i < N; ++i) {
        REQUIRE(received[i] == i);
    }
}

// ── Batch + zero-copy API ────────────────────────────────────────────────────

TEST_CASE("RingBuffer: push_bulk respects capacity and preserves order", "[ring_buffer]") {
    RingBuffer<GameEvent, 8> buf;          // holds 7
    std::vector<GameEvent> in;
    for (int i = 0; i < 10; ++i) in.push_back(make_event("bulk", 0.10f * static_cast<float>(i)));

    // Short write: asks for 10, capacity allows 7.
    REQUIRE(buf.push_bulk(in.data(), in.size()) == 7);
    REQUIRE(buf.push_bulk(in.data(), 1) == 0);      // now full

    for (int i = 0; i < 7; ++i) {
        GameEvent out{};
        REQUIRE(buf.pop(out));
        REQUIRE(out.win_probability == Approx(0.10f * static_cast<float>(i)).epsilon(1e-5f));
    }
    REQUIRE(buf.empty());
}

TEST_CASE("RingBuffer: push_bulk wraps correctly", "[ring_buffer]") {
    RingBuffer<GameEvent, 8> buf;
    std::vector<GameEvent> in;
    for (int i = 0; i < 5; ++i) in.push_back(make_event("w", 0.10f * static_cast<float>(i)));

    // Drive head past the wrap point, then bulk-push across it.
    for (int round = 0; round < 4; ++round) {
        REQUIRE(buf.push_bulk(in.data(), 5) == 5);
        for (int i = 0; i < 5; ++i) {
            GameEvent out{};
            REQUIRE(buf.pop(out));
            REQUIRE(out.win_probability == Approx(0.10f * static_cast<float>(i)).epsilon(1e-5f));
        }
    }
}

TEST_CASE("RingBuffer: push_bulk and pop_bulk round-trip", "[ring_buffer]") {
    RingBuffer<GameEvent, 16> buf;
    std::vector<GameEvent> in;
    for (int i = 0; i < 12; ++i) in.push_back(make_event("rt", 0.05f * static_cast<float>(i)));
    REQUIRE(buf.push_bulk(in.data(), 12) == 12);

    std::vector<GameEvent> out(12);
    REQUIRE(buf.pop_bulk(out.data(), 12) == 12);
    for (int i = 0; i < 12; ++i)
        REQUIRE(out[i].win_probability == Approx(0.05f * static_cast<float>(i)).epsilon(1e-5f));
    REQUIRE(buf.pop_bulk(out.data(), 12) == 0);
}

TEST_CASE("RingBuffer: claim/commit writes in place", "[ring_buffer]") {
    RingBuffer<GameEvent, 4> buf;          // holds 3

    for (int i = 0; i < 3; ++i) {
        GameEvent* slot = buf.claim();
        REQUIRE(slot != nullptr);
        *slot = make_event("zc", 0.20f * static_cast<float>(i));
        buf.commit();
    }
    REQUIRE(buf.claim() == nullptr);       // full: claim must refuse

    for (int i = 0; i < 3; ++i) {
        GameEvent out{};
        REQUIRE(buf.pop(out));
        REQUIRE(out.win_probability == Approx(0.20f * static_cast<float>(i)).epsilon(1e-5f));
    }
}

TEST_CASE("RingBuffer: claim without commit publishes nothing", "[ring_buffer]") {
    RingBuffer<GameEvent, 4> buf;
    GameEvent* slot = buf.claim();
    REQUIRE(slot != nullptr);
    *slot = make_event("uncommitted", 0.99f);

    // Not committed -> consumer must not see it.
    REQUIRE(buf.empty());
    GameEvent out{};
    REQUIRE_FALSE(buf.pop(out));

    // Claiming again returns the same slot, not a second one.
    REQUIRE(buf.claim() == slot);
    buf.commit();
    REQUIRE(buf.pop(out));
    REQUIRE(out.win_probability == Approx(0.99f).epsilon(1e-5f));
    REQUIRE(buf.empty());
}

TEST_CASE("RingBuffer: peek/consume reads in place", "[ring_buffer]") {
    RingBuffer<GameEvent, 4> buf;
    REQUIRE(buf.peek() == nullptr);        // empty

    REQUIRE(buf.push(make_event("p0", 0.31f)));
    REQUIRE(buf.push(make_event("p1", 0.32f)));

    const GameEvent* a = buf.peek();
    REQUIRE(a != nullptr);
    REQUIRE(a->win_probability == Approx(0.31f).epsilon(1e-5f));
    REQUIRE(buf.peek() == a);              // peek is non-destructive
    buf.consume();

    const GameEvent* b = buf.peek();
    REQUIRE(b != nullptr);
    REQUIRE(b->win_probability == Approx(0.32f).epsilon(1e-5f));
    buf.consume();

    REQUIRE(buf.peek() == nullptr);
    REQUIRE(buf.empty());
}

TEST_CASE("RingBuffer: zero-copy path is FIFO across wraparound", "[ring_buffer]") {
    RingBuffer<GameEvent, 4> buf;
    int produced = 0, consumed = 0;
    // More rounds than capacity so indices wrap repeatedly.
    for (int round = 0; round < 20; ++round) {
        while (GameEvent* slot = buf.claim()) {
            *slot = make_event("wrap", 0.001f * static_cast<float>(produced));
            buf.commit();
            ++produced;
        }
        while (const GameEvent* ev = buf.peek()) {
            REQUIRE(ev->win_probability ==
                    Approx(0.001f * static_cast<float>(consumed)).epsilon(1e-4f));
            buf.consume();
            ++consumed;
        }
    }
    REQUIRE(produced == consumed);
    REQUIRE(produced == 20 * 3);           // 3 live slots per round
}

TEST_CASE("RingBuffer: zero-copy producer to copying consumer, threaded", "[ring_buffer]") {
    RingBuffer<GameEvent, 1024> buf;
    constexpr int N = 50'000;
    std::vector<int> received;
    received.reserve(N);

    std::thread producer([&]() {
        for (int i = 0; i < N; ++i) {
            GameEvent* slot = nullptr;
            while ((slot = buf.claim()) == nullptr) { /* spin on full */ }
            *slot = make_event("zc_thread",
                               static_cast<float>(i) / static_cast<float>(N));
            buf.commit();
        }
    });
    std::thread consumer([&]() {
        for (int i = 0; i < N; ++i) {
            const GameEvent* ev = nullptr;
            while ((ev = buf.peek()) == nullptr) { /* spin on empty */ }
            received.push_back(
                static_cast<int>(ev->win_probability * static_cast<float>(N) + 0.5f));
            buf.consume();
        }
    });
    producer.join();
    consumer.join();

    REQUIRE(static_cast<int>(received.size()) == N);
    for (int i = 0; i < N; ++i) REQUIRE(received[i] == i);
}

// ── Benchmark ────────────────────────────────────────────────────────────────
// Run with: ./run_tests "[!benchmark]"  (benchmarks excluded by default)
// Target: < 200 ns per push+pop round-trip

TEST_CASE("RingBuffer: push+pop latency benchmark", "[ring_buffer][!benchmark]") {
    RingBuffer<GameEvent, 1024> buf;
    auto ev = make_event("bench", 0.65f);

    BENCHMARK("push + pop round-trip") {
        buf.push(ev);
        GameEvent out{};
        buf.pop(out);
        return out.win_probability;   // prevent dead-code elimination
    };
}
