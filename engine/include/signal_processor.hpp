/**
 * engine/include/signal_processor.hpp
 *
 * SignalProcessor — consumer side of the SPSC ring buffer.
 *
 * Threading model:
 *   Runs in a dedicated consumer thread. Pops GameEvents from the ring buffer,
 *   computes fractional Kelly sizing, checks exposure limits, and emits a
 *   BetRecommendation via a user-supplied callback.
 *
 *   Kelly and ExposureManager work is single-threaded within this consumer,
 *   so no additional locking is needed here (ExposureManager has its own
 *   internal mutex for concurrent callers from other threads, e.g. REST handlers).
 */
#pragma once

#include "game_event.hpp"
#include "ring_buffer.hpp"
#include "kelly_sizer.hpp"
#include "exposure_manager.hpp"
#include <functional>
#include <cstring>

namespace gridiron {

struct BetRecommendation {
    char   player_id[32];
    char   game_id[32];
    double recommended_bet_fraction;   // fraction of bankroll (after cap)
    double kelly_full;                 // full Kelly f* (pre-fractional, for logging)
    bool   approved;                   // false if ExposureManager rejected
};

using RecommendationCallback = std::function<void(const BetRecommendation&)>;

class SignalProcessor {
public:
    using Buffer = RingBuffer<GameEvent, 1024>;

    explicit SignalProcessor(Buffer&          ring_buf,
                             KellySizer&      kelly,
                             ExposureManager& exposure,
                             double           bankroll = 1000.0) noexcept
        : ring_buf_(ring_buf), kelly_(kelly), exposure_(exposure), bankroll_(bankroll) {}

    /**
     * process_one — Pop one event and produce a BetRecommendation.
     *
     * Returns false if the buffer is empty (nothing to process).
     * Invokes on_recommendation_ (if set) with the resulting recommendation.
     */
    bool process_one();

    /** Set callback invoked for every processed event. */
    void set_callback(RecommendationCallback cb) { on_recommendation_ = std::move(cb); }

private:
    Buffer&                ring_buf_;
    KellySizer&            kelly_;
    ExposureManager&       exposure_;
    double                 bankroll_;
    RecommendationCallback on_recommendation_;
};

}  // namespace gridiron
