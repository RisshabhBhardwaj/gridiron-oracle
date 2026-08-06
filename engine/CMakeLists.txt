cmake_minimum_required(VERSION 3.20)
project(gridiron_engine CXX)

# ── Language standard ──────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)        # -std=c++17 not -std=gnu++17

# Emit compile_commands.json for clangd / IDE integration.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── macOS stdlib fix (Apple clang 17 CommandLineTools incomplete libc++) ──────
# Apple clang 17 ships an incomplete C++ stdlib stub at:
#   /Library/Developer/CommandLineTools/usr/include/c++/v1/  (only __* internals)
# The full public libc++ headers live inside the Xcode SDK.
# Set via CMAKE_CXX_FLAGS (not add_compile_options) so the fix propagates
# to all FetchContent sub-projects (Catch2, nlohmann/json) in this build.
# MUST be declared before FetchContent_MakeAvailable calls.
if(APPLE)
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE MACOS_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(MACOS_SDK_PATH)
    set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -I${MACOS_SDK_PATH}/usr/include/c++/v1")
  endif()
endif()

# ── Threads (pthreads — producer/consumer threads for ring buffer) ─────────────
find_package(Threads REQUIRED)

# ── Dependencies via FetchContent ──────────────────────────────────────────────
include(FetchContent)

# nlohmann/json — IPC bridge JSON serialization + main.cpp config loading
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3          # pinned — change consciously, update this comment
  GIT_SHALLOW    TRUE
)

# Catch2 v3 — unit tests + BENCHMARK macro
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.2           # pinned — change consciously, update this comment
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(nlohmann_json Catch2)

# ── Boost (Beast + ASIO — required by ws_consumer) ─────────────────────────────
# Optional: install Boost to enable live WebSocket feed and ws_integration_tests.
#   macOS:  brew install boost
#   Ubuntu: sudo apt-get install libboost-all-dev
# Boost.Beast and Boost.Asio are header-only since Boost 1.69.
# Use Boost::headers target (always available when Boost is found).
find_package(Boost 1.82)

# ── Common compile options (applied AFTER FetchContent so they don't pollute deps)
add_compile_options(-Wall -Wextra -O2)

# ── Main engine executable ─────────────────────────────────────────────────────
add_executable(engine
  src/main.cpp
  src/ws_consumer.cpp
  src/signal_processor.cpp
  src/ipc_bridge.cpp
)
target_include_directories(engine PRIVATE include)
target_link_libraries(engine PRIVATE
  Threads::Threads
  nlohmann_json::nlohmann_json
)
if(Boost_FOUND)
  target_link_libraries(engine PRIVATE Boost::headers)
  target_compile_definitions(engine PRIVATE HAVE_BOOST)
endif()

# ── Unified test runner: run_tests ─────────────────────────────────────────────
# All four test suites compiled into one binary so `./run_tests` runs everything.
add_executable(run_tests
  tests/ring_buffer_test.cpp
  tests/kelly_test.cpp
  tests/exposure_manager_test.cpp
  tests/ipc_bridge_test.cpp
  tests/signal_processor_test.cpp
  tests/simd_mc_test.cpp
  tests/drive_mcmc_test.cpp
  src/ipc_bridge.cpp           # IpcBridge::parse_event / serialize_recommendation
  src/signal_processor.cpp     # SignalProcessor::process_one
  src/simd_mc.cpp
  src/drive_mcmc.cpp
)
target_include_directories(run_tests PRIVATE include)
target_link_libraries(run_tests PRIVATE
  Catch2::Catch2WithMain
  Threads::Threads
  nlohmann_json::nlohmann_json
)

# ── CTest integration (Catch2 auto-discovers TEST_CASE names) ─────────────────
enable_testing()
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(run_tests)

# ── Optional: ws_integration_tests (requires Boost) ───────────────────────────
if(Boost_FOUND)
  add_executable(ws_integration_tests
    tests/ws_consumer_test.cpp
    src/ws_consumer.cpp
    src/ipc_bridge.cpp
  )
  target_include_directories(ws_integration_tests PRIVATE include)
  target_compile_definitions(ws_integration_tests PRIVATE HAVE_BOOST)
  target_link_libraries(ws_integration_tests PRIVATE
    Catch2::Catch2WithMain
    Threads::Threads
    nlohmann_json::nlohmann_json
    Boost::headers
  )
  catch_discover_tests(ws_integration_tests)
endif()

# ── libFuzzer target (Clang only — enable with -DFUZZING=ON) ──────────────────
# Requires: clang + compiler-rt (ships with Xcode on macOS; apt install clang on Ubuntu)
# Usage:
#   cmake -B engine/build engine/ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DFUZZING=ON
#   cmake --build engine/build --target ring_buffer_fuzz
#   ./engine/build/ring_buffer_fuzz -max_total_time=60
option(FUZZING "Build libFuzzer harnesses (Clang only)" OFF)
if(FUZZING)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "FUZZING=ON requires Clang (current compiler: ${CMAKE_CXX_COMPILER_ID})")
  endif()
  add_executable(ring_buffer_fuzz fuzz/ring_buffer_fuzz.cpp)
  target_include_directories(ring_buffer_fuzz PRIVATE include)
  target_compile_options(ring_buffer_fuzz PRIVATE
    -fsanitize=fuzzer,address,undefined
    -fno-omit-frame-pointer
    -g
  )
  target_link_options(ring_buffer_fuzz PRIVATE
    -fsanitize=fuzzer,address,undefined
  )
  message(STATUS "libFuzzer target 'ring_buffer_fuzz' enabled")
endif()

# ── gridiron_engine_lib — Python ctypes shared library ────────────────────────
# Exposes season_sim_*, simd_*, drive_mcmc_* extern "C" API.
# Python: lib = ctypes.CDLL("engine/build/libgridiron_engine.dylib")
# Compile: cd engine && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
add_library(gridiron_engine_lib SHARED
  src/season_simulator.cpp
  src/simd_mc.cpp
  src/drive_mcmc.cpp
)
target_include_directories(gridiron_engine_lib PUBLIC include)
target_compile_options(gridiron_engine_lib PRIVATE -O3 -ffast-math)
set_target_properties(gridiron_engine_lib PROPERTIES
  OUTPUT_NAME "gridiron_engine"
  SOVERSION 1
)

# ARM NEON (Apple Silicon M1/M2/M3)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  target_compile_definitions(gridiron_engine_lib PRIVATE GRIDIRON_NEON=1)
  target_compile_options(gridiron_engine_lib PRIVATE -march=native)
  message(STATUS "gridiron_engine_lib: ARM NEON enabled")
else()
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
  if(COMPILER_SUPPORTS_AVX2)
    target_compile_definitions(gridiron_engine_lib PRIVATE GRIDIRON_AVX2=1)
    target_compile_options(gridiron_engine_lib PRIVATE -mavx2)
    message(STATUS "gridiron_engine_lib: AVX2 enabled")
  else()
    target_compile_definitions(gridiron_engine_lib PRIVATE GRIDIRON_SCALAR=1)
  endif()
endif()
