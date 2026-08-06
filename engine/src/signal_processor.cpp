/**
 * engine/src/signal_processor.cpp
 *
 * Consumer-side processing: pop a GameEvent, compute Kelly sizing,
 * check exposure limits, emit BetRecommendation via callback.
 */
#include "signal_processor.hpp"

namespace gridiron {

bool SignalProcessor::process_one() {
    GameEvent event{};
    if (!ring_buf_.pop(event)) {
        return false;  // buffer empty — caller should yield or spin-wait
    }

    // Full Kelly fraction (before fractional multiplier) — stored for logging.
    // f* = (b*p - q) / b, where b = decimal_odds - 1
    const double b      = static_cast<double>(event.decimal_odds) - 1.0;
    const double p      = static_cast<double>(event.win_probability);
    const double f_star = (b > 0.0) ? (b * p - (1.0 - p)) / b : 0.0;

    // Fractional Kelly with hard cap.
    const double recommended = kelly_.size_bet(
        static_cast<double>(event.win_probability),
        static_cast<double>(event.decimal_odds)
    );

    BetRecord record{
        std::string(event.player_id),
        std::string(event.game_id),
        std::string(event.position),   // position parsed from JSON payload
        recommended * bankroll_,
    };

    const auto decision = exposure_.try_place_bet(record);

    BetRecommendation rec{};
    std::strncpy(rec.player_id, event.player_id, sizeof(rec.player_id) - 1);
    rec.player_id[sizeof(rec.player_id) - 1] = '\0';   // guarantee null termination
    std::strncpy(rec.game_id,   event.game_id,   sizeof(rec.game_id)   - 1);
    rec.game_id[sizeof(rec.game_id) - 1]     = '\0';   // guarantee null termination
    rec.recommended_bet_fraction = recommended;
    rec.kelly_full               = f_star;
    rec.approved                 = (decision == ExposureDecision::APPROVED);

    if (on_recommendation_) {
        on_recommendation_(rec);
    }

    return true;
}

}  // namespace gridiron
