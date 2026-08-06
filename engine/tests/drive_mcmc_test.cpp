/**
 * engine/tests/drive_mcmc_test.cpp
 *
 * Catch2 v3 unit tests for DriveMCMC and its C API wrappers.
 *
 * Tests cover deterministic policy boundaries, transition loading, output
 * invariants, and basic C API behavior.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cstdio>
#include <fstream>
#include <string>

#include "drive_mcmc.hpp"

using namespace gridiron;
using Catch::Approx;

TEST_CASE("DriveMCMC: fourth down decision follows current threshold policy",
          "[drive_mcmc]") {
    DriveMCMC sim(32, 42);

    REQUIRE(sim.fourth_down_decision(70, 5) == DriveOutcome::FIELD_GOAL);
    REQUIRE(sim.fourth_down_decision(70, 6) == DriveOutcome::TURNOVER_DOWNS);
    REQUIRE(sim.fourth_down_decision(50, 2) == DriveOutcome::IN_PROGRESS);
    REQUIRE(sim.fourth_down_decision(50, 3) == DriveOutcome::PUNT);
    REQUIRE(sim.fourth_down_decision(30, 1) == DriveOutcome::PUNT);
}

TEST_CASE("DriveMCMC: simulate_drive returns a valid probability distribution",
          "[drive_mcmc]") {
    DriveMCMC sim(2000, 42);
    const DriveState start{25, 1, 10, DriveOutcome::IN_PROGRESS};

    const DriveResult res = sim.simulate_drive(start);

    REQUIRE(res.p_touchdown >= 0.0f);
    REQUIRE(res.p_field_goal >= 0.0f);
    REQUIRE(res.p_punt >= 0.0f);
    REQUIRE(res.p_turnover >= 0.0f);
    REQUIRE(res.p_touchdown <= 1.0f);
    REQUIRE(res.p_field_goal <= 1.0f);
    REQUIRE(res.p_punt <= 1.0f);
    REQUIRE(res.p_turnover <= 1.0f);
    REQUIRE(res.p_touchdown + res.p_field_goal + res.p_punt + res.p_turnover
            == Approx(1.0f).margin(1e-5));
    REQUIRE(res.drive_value
            == Approx(7.0f * res.p_touchdown + 3.0f * res.p_field_goal).margin(1e-6));
    REQUIRE(res.p50_yards <= res.p90_yards);
    REQUIRE(res.n_simulations == 2000);
}

TEST_CASE("DriveMCMC: max_plays zero forces timeout punt with zero net yards",
          "[drive_mcmc]") {
    DriveMCMC sim(128, 42);
    sim.set_max_plays(0);
    const DriveState start{25, 1, 10, DriveOutcome::IN_PROGRESS};

    const DriveResult res = sim.simulate_drive(start);

    REQUIRE(res.p_punt == Approx(1.0f).margin(1e-7));
    REQUIRE(res.p_touchdown == Approx(0.0f).margin(1e-7));
    REQUIRE(res.p_field_goal == Approx(0.0f).margin(1e-7));
    REQUIRE(res.p_turnover == Approx(0.0f).margin(1e-7));
    REQUIRE(res.expected_yards == Approx(0.0f).margin(1e-7));
    REQUIRE(res.p50_yards == Approx(0.0f).margin(1e-7));
    REQUIRE(res.p90_yards == Approx(0.0f).margin(1e-7));
}

TEST_CASE("DriveMCMC: CSV transition override can force deterministic touchdown",
          "[drive_mcmc]") {
    const std::string path = "drive_mcmc_test_transitions.csv";
    {
        std::ofstream out(path);
        REQUIRE(out.is_open());
        out << "field_pos_bucket,down,ytg_bucket,mean_gain,gain_std,p_turnover,p_penalty_gain,p_penalty_loss\n";
        out << "0,0,0,99,0,0,0,0\n";
    }

    DriveMCMC sim(64, 42);
    REQUIRE(sim.load_transitions_from_csv(path));

    const DriveState start{5, 1, 1, DriveOutcome::IN_PROGRESS};
    const DriveResult res = sim.simulate_drive(start);

    std::remove(path.c_str());

    REQUIRE(res.p_touchdown == Approx(1.0f).margin(1e-7));
    REQUIRE(res.p_field_goal == Approx(0.0f).margin(1e-7));
    REQUIRE(res.p_punt == Approx(0.0f).margin(1e-7));
    REQUIRE(res.p_turnover == Approx(0.0f).margin(1e-7));
    REQUIRE(res.expected_yards == Approx(95.0f).margin(1e-6));
}

TEST_CASE("DriveMCMC: missing transition CSV returns false",
          "[drive_mcmc]") {
    DriveMCMC sim(16, 42);
    REQUIRE_FALSE(sim.load_transitions_from_csv("definitely_missing_drive_transitions.csv"));
}

TEST_CASE("DriveMCMC C API: simulate writes sane outputs",
          "[drive_mcmc][c_api]") {
    void* handle = drive_mcmc_create(512, 7);
    REQUIRE(handle != nullptr);

    drive_mcmc_load_defaults(handle);

    float p_td = -1.0f;
    float p_fg = -1.0f;
    float expected_yards = 0.0f;
    float drive_value = 0.0f;
    drive_mcmc_simulate(handle, 25, 1, 10, &p_td, &p_fg, &expected_yards, &drive_value);

    REQUIRE(p_td >= 0.0f);
    REQUIRE(p_td <= 1.0f);
    REQUIRE(p_fg >= 0.0f);
    REQUIRE(p_fg <= 1.0f);
    REQUIRE(drive_value == Approx(7.0f * p_td + 3.0f * p_fg).margin(1e-6));
    REQUIRE(expected_yards == expected_yards); // not NaN

    drive_mcmc_destroy(handle);
}
