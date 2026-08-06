/**
 * engine/fuzz/ring_buffer_fuzz.cpp
 *
 * libFuzzer harness for RingBuffer<uint32_t, 16>.
 *
 * Build (Clang only — libFuzzer is not available with GCC):
 *   cmake -B engine/build engine/ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DFUZZING=ON
 *   cmake --build engine/build --target ring_buffer_fuzz
 *
 * Run:
 *   ./engine/build/ring_buffer_fuzz -max_total_time=60
 *   ./engine/build/ring_buffer_fuzz corpus/      # seed corpus
 *
 * What this tests:
 *   - The fuzzer drives a sequence of push/pop operations whose order and
 *     values are derived from the input bytes. Any crash, ASAN/UBSAN report,
 *     or violated SPSC invariant (FIFO order, no item loss, no phantom reads)
 *     is flagged as a finding.
 *
 * Invariants checked per iteration:
 *   1. If push() returns true, the item is enqueued (tracked in a reference
 *      std::queue<uint32_t> that mirrors the expected state).
 *   2. If pop() returns true, the item must match the front of the reference
 *      queue — FIFO order is preserved.
 *   3. empty() is consistent with the reference queue's size.
 */

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <queue>

#include "ring_buffer.hpp"

using namespace gridiron;

// Small capacity to stress the full/empty boundary conditions quickly.
static constexpr std::size_t kCapacity = 16;
using Buf = RingBuffer<uint32_t, kCapacity>;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    Buf buf;
    std::queue<uint32_t> reference;

    for (std::size_t i = 0; i + 1 < size; i += 2) {
        const uint8_t op  = data[i] & 0x01;   // bit 0: 0 = push, 1 = pop
        const uint8_t val = data[i + 1];       // payload for push

        if (op == 0) {
            // Push: if the buffer accepts the item, mirror in reference.
            const bool pushed = buf.push(static_cast<uint32_t>(val));
            if (pushed) {
                reference.push(static_cast<uint32_t>(val));
            }
        } else {
            // Pop: item must match front of reference queue (FIFO invariant).
            uint32_t item = 0;
            const bool popped = buf.pop(item);
            if (popped) {
                assert(!reference.empty() && "pop() succeeded but reference queue is empty");
                assert(item == reference.front() && "FIFO order violated");
                reference.pop();
            } else {
                assert(reference.empty() && "pop() failed but reference queue is non-empty");
            }
        }

        // empty() must agree with reference queue.
        assert(buf.empty() == reference.empty() && "empty() inconsistent with reference");
    }

    return 0;
}
