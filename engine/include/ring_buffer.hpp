/**
 * engine/include/ring_buffer.hpp
 *
 * Lock-free single-producer single-consumer (SPSC) ring buffer.
 *
 * Memory ordering rationale (CLAUDE.md §3 C++ Rules / Interview anchor §1):
 *
 *   head_ is the write index, owned exclusively by the producer.
 *   tail_ is the read  index, owned exclusively by the consumer.
 *
 *   push() — producer thread:
 *     1. Load head_ with relaxed  — self-read; no cross-thread sync needed.
 *     2. Load tail_ with acquire  — establishes happens-before with the
 *        consumer's release store on tail_, ensuring we see freed slots.
 *     3. Write buffer_[head_].
 *     4. Store head_ with release — publishes the new item; pairs with
 *        the consumer's acquire load in pop().
 *
 *   pop() — consumer thread:
 *     1. Load tail_ with relaxed  — self-read; no cross-thread sync needed.
 *     2. Load head_ with acquire  — establishes happens-before with the
 *        producer's release store on head_, ensuring we see written items.
 *     3. Read buffer_[tail_].
 *     4. Store tail_ with release — frees the slot; pairs with the
 *        producer's acquire load in push().
 *
 *   Result: exactly two acquire/release pairs per round-trip, zero seq_cst
 *   fences. This is the minimum possible synchronization for correct SPSC.
 *
 * Design constraints (CLAUDE.md §3):
 *   - Capacity must be a power of 2 (enforced by static_assert); enables
 *     fast bitmasking instead of modulo.
 *   - head_ and tail_ each live in a 64-byte aligned cache-line-sized struct
 *     to prevent false sharing between producer and consumer.
 *   - Zero heap allocation after construction.
 */
#pragma once

#include <atomic>
#include <array>
#include <cstddef>

namespace gridiron {

template <typename T, std::size_t Capacity>
class RingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of 2");
    static_assert(Capacity >= 2, "Capacity must be at least 2");

public:
    // PaddedAtomic: wraps one atomic index in exactly one 64-byte cache line.
    // alignas(64) prevents false sharing between producer (head_) and consumer (tail_).
    // The padding array fills the remainder of the cache line so the struct
    // occupies exactly 64 bytes, regardless of sizeof(atomic<size_t>).
    struct alignas(64) PaddedAtomic {
        std::atomic<std::size_t> val{0};
        // On x86-64, sizeof(atomic<size_t>) == 8. Padding = 64 - 8 = 56 bytes.
        char pad[64 - sizeof(std::atomic<std::size_t>)];
    };

    RingBuffer() = default;

    // Non-copyable, non-movable (atomics are not copyable; buffer may be large).
    RingBuffer(const RingBuffer&)            = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    /**
     * push — Producer thread only.
     *
     * Enqueues `item` at the current write position.
     * Returns true on success; false if the buffer is full (caller must retry).
     */
    bool push(const T& item) noexcept {
        // relaxed: producer reading its own index — no cross-thread sync needed.
        const std::size_t h      = head_.val.load(std::memory_order_relaxed);
        const std::size_t next_h = (h + 1) & kMask;

        // acquire: sync with consumer's release store on tail_ — ensures we
        // observe the consumer's most recent slot reclamation before checking fullness.
        if (next_h == tail_.val.load(std::memory_order_acquire)) {
            // Buffer full — returns false. Caller (ws_consumer.cpp) logs the drop
            // and retries with exponential backoff (1 s → 30 s cap). No blocking.
            return false;
        }

        buffer_[h] = item;

        // release: makes the written item and all stores before it visible to
        // the consumer before the consumer can observe the updated head_.
        head_.val.store(next_h, std::memory_order_release);
        return true;
    }

    /**
     * pop — Consumer thread only.
     *
     * Dequeues the front item into `item`.
     * Returns true on success; false if the buffer is empty.
     */
    bool pop(T& item) noexcept {
        // relaxed: consumer reading its own index — no cross-thread sync needed.
        const std::size_t t = tail_.val.load(std::memory_order_relaxed);

        // acquire: sync with producer's release store on head_ — ensures we
        // observe the item written by push() before reading buffer_[tail_].
        if (t == head_.val.load(std::memory_order_acquire)) {
            return false;  // buffer empty
        }

        item = buffer_[t];

        // release: frees the slot — makes the reclamation visible to the
        // producer before the producer can observe the updated tail_.
        tail_.val.store((t + 1) & kMask, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool empty() const noexcept {
        // relaxed: empty() is advisory only — pop() re-checks with full acquire
        // semantics, so we need no cross-thread synchronization here.
        // Using acquire for both self-reads would over-synchronize without
        // providing additional correctness guarantees.
        return tail_.val.load(std::memory_order_relaxed)
            == head_.val.load(std::memory_order_relaxed);
    }

    static constexpr std::size_t capacity() noexcept { return Capacity; }

private:
    static constexpr std::size_t kMask = Capacity - 1;

    PaddedAtomic            head_{};       // producer-owned write index
    std::array<T, Capacity> buffer_{};     // fixed-size storage — no heap alloc
    PaddedAtomic            tail_{};       // consumer-owned read index
};

}  // namespace gridiron
