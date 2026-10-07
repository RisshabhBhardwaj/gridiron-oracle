/**
 * bench/spsc_latency_bench.cpp
 *
 * Open-loop latency + saturation throughput bench:
 * lock-free SPSC ring vs mutex + std::queue, at matched batch sizes.
 *
 * WHY THIS FILE LOOKS THE WAY IT DOES
 *
 * Coordinated omission. Message i is due at t0 + i*period regardless of
 * whether the producer got there on time. Latency is measured from `due`,
 * not from the actual push, so a producer stall shows up in the tail
 * instead of silently vanishing.
 *
 * Latency decomposition. Reporting one end-to-end number cannot tell you
 * whether a millisecond tail came from the queue or from the OS descheduling
 * the generator. So every message carries two stamps and we report:
 *     end-to-end     = consumer_done - due
 *     prod lateness  = actual_push   - due     (generator scheduling)
 *     transport      = consumer_done - actual_push  (the queue itself)
 * On this machine the tail lives almost entirely in `prod lateness`.
 *
 * Preemption accounting. Both spin loops watch for their own clock jumping
 * more than kGapThresholdNs between consecutive reads, which is a direct
 * observation of this thread losing the core. Reported per config alongside
 * getrusage() involuntary context switches, so a reader can see how much of
 * the tail the scheduler owns.
 *
 * Histogram, not an array. The old version allocated rate*duration int64s
 * (560 MB at 1M msg/s x 60 s), faulted it, then sorted 70M elements. This
 * uses an HDR-style log-linear histogram: 1 MB fixed, <=0.05% relative
 * error, so high rates are actually reachable.
 *
 * Repetitions, interleaved. One run per cell is how the previous results
 * ended up non-monotonic (100k p99 worse than 500k p99 -- variance, not
 * signal). Every cell runs --reps times, we report the median and the
 * full [min-max] spread, and configs are interleaved rather than run in
 * blocks so thermal drift hits every variant equally instead of becoming
 * a bias in favour of whichever ran first.
 *
 * No cherry-picked headline. The old file scanned every rate for the most
 * favourable SPSC/mutex ratio and printed that one. This prints the whole
 * sweep and leaves the reading to the reader.
 *
 * Clock. mach_absolute_time only; ticks converted once at the end. Note the
 * timebase quantum is printed -- on Apple silicon it is ~41.67 ns, so a p50
 * quoted to the nanosecond is false precision and the output says so.
 *
 * QoS. pthread_attr_set_qos_class_np before create (not after the thread
 * already landed on an E-core), and never mixed with Mach affinity policies,
 * which reset the QoS class to UNSPECIFIED.
 */
// _GNU_SOURCE must be defined before any system header, or glibc will not
// declare pthread_setaffinity_np and Linux core pinning silently disappears.
// g++ defines it for C++ already; clang++ with libstdc++ does not always.
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif

#include "ring_buffer.hpp"
#include "game_event.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#include <pthread.h>
#endif
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>
#endif
#include <sys/resource.h>

using gridiron::GameEvent;
using gridiron::RingBuffer;

namespace {

constexpr std::size_t kCapacity = 1024;              // matches engine/src/main.cpp
constexpr std::size_t kQueueCap = kCapacity - 1;     // ring keeps one sentinel slot
constexpr std::size_t kBatch    = 32;                // bulk drain size
constexpr std::int64_t kGapThresholdNs = 20000;      // 20us: far above any coherence event

// ── HDR-style log-linear histogram ───────────────────────────────────────────
class Histogram {
public:
    static constexpr int       kSigBits = 11;                 // <=0.05% rel. error
    static constexpr std::uint64_t kSub     = 1ull << kSigBits;
    static constexpr std::uint64_t kLinear  = 2 * kSub;        // exact below this
    static constexpr std::uint64_t kBuckets = kLinear + 62 * kSub;

    Histogram() : counts_(kBuckets, 0) {}

    void record(std::int64_t v) noexcept {
        if (v < 0) v = 0;                       // clock skew guard; never negative-bias
        const std::uint64_t u = static_cast<std::uint64_t>(v);
        ++counts_[index(u)];
        ++total_;
        if (u > max_) max_ = u;
    }
    void reset() noexcept {
        std::fill(counts_.begin(), counts_.end(), 0);
        total_ = 0; max_ = 0;
    }
    std::uint64_t total() const noexcept { return total_; }
    std::uint64_t max()   const noexcept { return max_; }

    std::uint64_t percentile(double p) const noexcept {
        if (total_ == 0) return 0;
        std::uint64_t target = static_cast<std::uint64_t>(p * static_cast<double>(total_));
        if (target >= total_) target = total_ - 1;
        std::uint64_t seen = 0;
        for (std::uint64_t i = 0; i < counts_.size(); ++i) {
            seen += counts_[i];
            if (seen > target) {
                const std::uint64_t hi = upper_bound(i);
                return hi < max_ ? hi : max_;
            }
        }
        return max_;
    }
private:
    static std::uint64_t index(std::uint64_t v) noexcept {
        if (v < kLinear) return v;
        const int           msb = 63 - __builtin_clzll(v);
        const std::uint64_t b   = static_cast<std::uint64_t>(msb) - kSigBits;
        return kLinear + (b - 1) * kSub + ((v >> b) - kSub);
    }
    static std::uint64_t upper_bound(std::uint64_t i) noexcept {
        if (i < kLinear) return i;
        const std::uint64_t j = i - kLinear, b = j / kSub + 1, sub = j % kSub;
        return ((kSub + sub + 1) << b) - 1;
    }
    std::vector<std::uint64_t> counts_;
    std::uint64_t total_ = 0, max_ = 0;
};

// ── Clock ────────────────────────────────────────────────────────────────────
// Both platforms expose the same three primitives: ticks_now(), ticks_to_ns()
// and a ticks-per-second constant, so nothing above this line is conditional.
#if defined(__APPLE__)
std::uint32_t g_numer = 1, g_denom = 1;
double        g_ns_per_tick = 1.0;
std::uint64_t g_ticks_per_sec = 1000000000ull;

void init_timebase() {
    mach_timebase_info_data_t info{};
    mach_timebase_info(&info);
    g_numer = info.numer; g_denom = info.denom;
    g_ns_per_tick   = static_cast<double>(info.numer) / static_cast<double>(info.denom);
    g_ticks_per_sec = static_cast<std::uint64_t>(1e9 * info.denom / info.numer);
}
inline std::uint64_t ticks_now() noexcept { return mach_absolute_time(); }
inline std::int64_t ticks_to_ns(std::int64_t t) noexcept {
    return static_cast<std::int64_t>(
        static_cast<unsigned __int128>(static_cast<std::uint64_t>(t)) * g_numer / g_denom);
}
const char* clock_name() { return "mach_absolute_time"; }

#elif defined(__linux__)
double        g_ns_per_tick   = 1.0;   // CLOCK_MONOTONIC ticks are nanoseconds
std::uint64_t g_ticks_per_sec = 1000000000ull;

void init_timebase() {
    // Report the real resolution rather than assuming 1 ns. On a normal
    // x86-64/aarch64 box with a vDSO this is 1 ns and the read is ~20 ns.
    timespec res{};
    if (clock_getres(CLOCK_MONOTONIC, &res) == 0) {
        const double r = static_cast<double>(res.tv_sec) * 1e9 + static_cast<double>(res.tv_nsec);
        if (r > 0.0) g_ns_per_tick = r;
    }
}
inline std::uint64_t ticks_now() noexcept {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);   // vDSO, no syscall
    return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull
         + static_cast<std::uint64_t>(ts.tv_nsec);
}
inline std::int64_t ticks_to_ns(std::int64_t t) noexcept { return t; }
const char* clock_name() { return "clock_gettime(CLOCK_MONOTONIC)"; }

#else
double        g_ns_per_tick   = 1.0;
std::uint64_t g_ticks_per_sec = 1000000000ull;
void init_timebase() {}
inline std::uint64_t ticks_now() noexcept {
    using C = std::chrono::steady_clock;
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(C::now().time_since_epoch()).count());
}
inline std::int64_t ticks_to_ns(std::int64_t t) noexcept { return t; }
const char* clock_name() { return "std::chrono::steady_clock"; }
#endif

// Exact open-loop schedule. Computing i*(ticks_per_sec/rate) would truncate the
// period once and let the error accumulate across tens of millions of messages;
// this keeps the division at full width so the schedule never drifts.
inline std::uint64_t due_ticks(std::uint64_t t0, std::uint64_t i, std::uint64_t rate) noexcept {
    return t0 + static_cast<std::uint64_t>(
        (static_cast<unsigned __int128>(i) * g_ticks_per_sec) / rate);
}

// ── Benchmark thread: P-core bias on macOS, hard core pinning on Linux ───────
//
// macOS cannot pin a thread to a core. QOS_CLASS_USER_INTERACTIVE biases the
// scheduler toward P-cores and that is the whole of the control available,
// which is why the macOS numbers carry a run-to-run placement spread that no
// amount of repetition removes.
//
// Linux can pin, and that is the point of running this there: with the
// producer and consumer on two dedicated cores (ideally isolcpus/nohz_full)
// the tail stops being a measurement of the scheduler.
int g_pin_producer = -1;      // CPU id, or -1 for unpinned
int g_pin_consumer = -1;
bool g_qos_ok      = true;    // macOS: QoS class actually applied
bool g_pin_ok      = true;    // Linux: affinity actually applied

bool pin_self_to_cpu(int cpu) {
#if defined(__linux__)
    if (cpu < 0) return true;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<unsigned>(cpu), &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)cpu;
    return true;   // no-op where pinning is unavailable
#endif
}

struct ThreadArg {
    std::function<void()>* fn;
    int cpu;
};

void* bench_thread_entry(void* raw) {
    auto* arg = static_cast<ThreadArg*>(raw);
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    qos_class_t q{}; int rel = 0;
    pthread_get_qos_class_np(pthread_self(), &q, &rel);
    if (q != QOS_CLASS_USER_INTERACTIVE) g_qos_ok = false;
#endif
    if (!pin_self_to_cpu(arg->cpu)) g_pin_ok = false;
    (*arg->fn)();
    return nullptr;
}

class BenchThread {
public:
    template <typename F>
    BenchThread(F&& f, int cpu) {
        fn_  = std::make_unique<std::function<void()>>(std::forward<F>(f));
        arg_ = std::make_unique<ThreadArg>(ThreadArg{fn_.get(), cpu});
        pthread_attr_t attr;
        pthread_attr_init(&attr);
#if defined(__APPLE__)
        // Set QoS on the attr *before* create, so the thread never lands on an
        // E-core first and gets migrated afterwards.
        pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
#endif
        if (pthread_create(&t_, &attr, bench_thread_entry, arg_.get()) != 0) {
            std::fprintf(stderr, "pthread_create failed\n");
            std::exit(1);
        }
        pthread_attr_destroy(&attr);
    }
    BenchThread(const BenchThread&) = delete;
    BenchThread& operator=(const BenchThread&) = delete;
    void join() { if (!joined_) { pthread_join(t_, nullptr); joined_ = true; } }
    ~BenchThread() { join(); }
private:
    pthread_t t_{};
    std::unique_ptr<std::function<void()>> fn_;
    std::unique_ptr<ThreadArg> arg_;
    bool joined_ = false;
};

// ── Legacy ring: the pre-fix implementation, kept as a control ───────────────
// Identical to gridiron::RingBuffer except that push()/pop() acquire-load the
// *opposite* index on every single operation instead of consulting a cached
// copy. This is what shipped before; it is compiled in so the before/after
// claim in docs/BENCHMARKS.md is reproducible from one command rather than being
// a number you have to take on trust.
template <typename T, std::size_t Capacity>
class alignas(gridiron::kDestructiveInterference) LegacyRingBuffer {
    static constexpr std::size_t kMask = Capacity - 1;
    static constexpr std::size_t kLine = gridiron::kDestructiveInterference;
    struct alignas(kLine) PaddedAtomic {
        std::atomic<std::size_t> val{0};
        char pad[kLine - sizeof(std::atomic<std::size_t>)];
    };
public:
    bool push(const T& item) noexcept {
        const std::size_t h = head_.val.load(std::memory_order_relaxed);
        const std::size_t next_h = (h + 1) & kMask;
        if (next_h == tail_.val.load(std::memory_order_acquire)) return false;
        buffer_[h] = item;
        head_.val.store(next_h, std::memory_order_release);
        return true;
    }
    bool pop(T& item) noexcept {
        const std::size_t t = tail_.val.load(std::memory_order_relaxed);
        if (t == head_.val.load(std::memory_order_acquire)) return false;
        item = buffer_[t];
        tail_.val.store((t + 1) & kMask, std::memory_order_release);
        return true;
    }
private:
    PaddedAtomic            head_{};
    std::array<T, Capacity> buffer_{};
    PaddedAtomic            tail_{};
};

// ── Queue adapters (uniform pop_batch interface) ─────────────────────────────
struct SpscAdapter {
    RingBuffer<GameEvent, kCapacity> q;
    static const char* name(bool bulk) { return bulk ? "lock-free SPSC (bulk 32)" : "lock-free SPSC"; }
    bool push(const GameEvent& e) noexcept { return q.push(e); }
    std::size_t pop_batch(GameEvent* out, std::size_t n, bool bulk) noexcept {
        if (bulk) return q.pop_bulk(out, n);
        return q.pop(out[0]) ? 1u : 0u;
    }
    void stop() noexcept {}
    static constexpr bool blocking = false;
};

struct LegacySpscAdapter {
    LegacyRingBuffer<GameEvent, kCapacity> q;
    bool push(const GameEvent& e) noexcept { return q.push(e); }
    std::size_t pop_batch(GameEvent* out, std::size_t, bool) noexcept {
        return q.pop(out[0]) ? 1u : 0u;   // no bulk API on the legacy ring
    }
    void stop() noexcept {}
    static constexpr bool blocking = false;
};

struct MutexAdapter {
    std::mutex mu; std::condition_variable cv_ne, cv_nf;
    std::queue<GameEvent> q; bool stopped = false;
    static const char* name(bool bulk) { return bulk ? "mutex + std::queue (bulk 32)" : "mutex + std::queue"; }
    bool push(const GameEvent& e) {
        std::unique_lock<std::mutex> lk(mu);
        cv_nf.wait(lk, [&] { return q.size() < kQueueCap || stopped; });
        if (stopped) return true;
        q.push(e); cv_ne.notify_one(); return true;
    }
    std::size_t pop_batch(GameEvent* out, std::size_t n, bool bulk) {
        std::unique_lock<std::mutex> lk(mu);
        cv_ne.wait(lk, [&] { return !q.empty() || stopped; });
        if (q.empty()) return 0;
        std::size_t k = 0;
        const std::size_t lim = bulk ? n : 1;
        while (k < lim && !q.empty()) { out[k++] = q.front(); q.pop(); }
        cv_nf.notify_one();
        return k;
    }
    void stop() {
        std::lock_guard<std::mutex> lk(mu);
        stopped = true; cv_ne.notify_all(); cv_nf.notify_all();
    }
    static constexpr bool blocking = true;
};

inline void consume_work(const GameEvent& e, volatile float* sink) noexcept {
    const float edge = e.win_probability - (1.0f / e.decimal_odds);
    *sink = edge * e.stat_delta;
}

GameEvent make_payload() {
    GameEvent ev{};
    std::strncpy(ev.player_id,  "00-0032765",     sizeof(ev.player_id)  - 1);
    std::strncpy(ev.game_id,    "2025_01_KC_LAC", sizeof(ev.game_id)    - 1);
    std::strncpy(ev.event_type, "projection",     sizeof(ev.event_type) - 1);
    std::strncpy(ev.position,   "QB",             sizeof(ev.position)   - 1);
    ev.stat_delta = 87.5f; ev.win_probability = 0.62f; ev.decimal_odds = 2.10f;
    ev.timestamp_ms = 0;
    return ev;
}

// ── Results ──────────────────────────────────────────────────────────────────
struct LatResult {
    double e2e_p50 = 0, e2e_p99 = 0, e2e_p999 = 0, e2e_max = 0;
    double late_p50 = 0, late_p99 = 0, late_p999 = 0;
    double tr_p50 = 0, tr_p99 = 0, tr_p999 = 0;
    std::uint64_t prod_gaps = 0, cons_gaps = 0;
    std::int64_t  worst_gap_ns = 0;
    long          nivcsw = 0;
    std::uint64_t push_retries = 0;
};

// Stamp-ring size. The producer's maximum lead over the *stamps the consumer
// has still to read* is not kQueueCap: a batching consumer pops up to kBatch
// messages into a local array and reads their stamps afterwards, by which time
// those slots are already free for the producer to refill. Worst case is
// therefore kQueueCap + kBatch (~1055). Sized to 4x that bound so a stamp can
// never be overwritten before it is read -- getting this wrong would silently
// corrupt the prod-lateness and transport columns, which is precisely what
// this bench exists to report.
constexpr std::size_t kStampSlots = 4096;
static_assert(kStampSlots > kQueueCap + kBatch, "stamp ring must exceed max producer lead");
constexpr std::size_t kStampMaskV = kStampSlots - 1;

/**
 * One open-loop run. The producer stamps push_ts[i & kStampMaskV] immediately
 * before pushing message i; the consumer reads it after popping message i.
 * The queue's own release/acquire pair supplies the happens-before edge.
 */
template <typename Adapter>
LatResult run_latency(std::uint64_t rate, double warmup_s, double duration_s, bool bulk) {
    const std::uint64_t n_warm = static_cast<std::uint64_t>(rate * warmup_s);
    const std::uint64_t n_meas = static_cast<std::uint64_t>(rate * duration_s);
    const std::uint64_t n_tot  = n_warm + n_meas;

    Adapter adapter;
    std::vector<std::uint64_t> push_ts(kStampSlots, 0);
    Histogram h_e2e, h_late, h_tr;
    std::uint64_t prod_gaps = 0, cons_gaps = 0, push_retries = 0;
    // One worst-gap per thread: a single shared variable would be a data race
    // on a number this file publishes.
    std::int64_t  prod_worst_gap = 0, cons_worst_gap = 0;
    volatile float sink = 0.0f;
    GameEvent proto = make_payload();

    rusage ru0{}; getrusage(RUSAGE_SELF, &ru0);

    BenchThread consumer([&] {
        std::vector<GameEvent> batch(kBatch);
        std::uint64_t i = 0;
        std::uint64_t last = ticks_now();
        while (i < n_tot) {
            const std::size_t k = adapter.pop_batch(batch.data(), kBatch, bulk);
            if (k == 0) {
                if (!Adapter::blocking) {
                    const std::uint64_t t = ticks_now();
                    const std::int64_t d = ticks_to_ns(static_cast<std::int64_t>(t - last));
                    if (d > kGapThresholdNs) {
                        ++cons_gaps;
                        if (d > cons_worst_gap) cons_worst_gap = d;
                    }
                    last = t;
                }
                continue;
            }
            for (std::size_t j = 0; j < k && i < n_tot; ++j, ++i) {
                consume_work(batch[j], &sink);
                const std::uint64_t done = ticks_now();
                if (i >= n_warm) {
                    const auto due  = static_cast<std::uint64_t>(batch[j].timestamp_ms);
                    const auto push = push_ts[i & kStampMaskV];
                    h_e2e.record(ticks_to_ns(static_cast<std::int64_t>(done - due)));
                    h_late.record(ticks_to_ns(static_cast<std::int64_t>(push - due)));
                    h_tr.record(ticks_to_ns(static_cast<std::int64_t>(done - push)));
                }
            }
            last = ticks_now();
        }
    }, g_pin_consumer);

    BenchThread producer([&] {
        const std::uint64_t t0 = ticks_now();
        for (std::uint64_t i = 0; i < n_tot; ++i) {
            const std::uint64_t due = due_ticks(t0, i, rate);
            proto.timestamp_ms = static_cast<std::int64_t>(due);
            // Busy-wait: mach_wait_until / nanosleep overshoot by several us on
            // macOS, which would become a systematic p50 floor measured from `due`.
            std::uint64_t last = ticks_now();
            while (last < due) {
                const std::uint64_t t = ticks_now();
                const std::int64_t d = ticks_to_ns(static_cast<std::int64_t>(t - last));
                if (d > kGapThresholdNs) {
                    ++prod_gaps;
                    if (d > prod_worst_gap) prod_worst_gap = d;
                }
                last = t;
            }
            push_ts[i & kStampMaskV] = ticks_now();
            while (!adapter.push(proto)) { ++push_retries; }
        }
        adapter.stop();
    }, g_pin_producer);

    producer.join();
    consumer.join();

    rusage ru1{}; getrusage(RUSAGE_SELF, &ru1);

    auto us = [](std::uint64_t ns) { return static_cast<double>(ns) / 1000.0; };
    LatResult r;
    r.e2e_p50 = us(h_e2e.percentile(0.50));  r.e2e_p99  = us(h_e2e.percentile(0.99));
    r.e2e_p999= us(h_e2e.percentile(0.999)); r.e2e_max  = us(h_e2e.max());
    r.late_p50= us(h_late.percentile(0.50)); r.late_p99 = us(h_late.percentile(0.99));
    r.late_p999=us(h_late.percentile(0.999));
    r.tr_p50  = us(h_tr.percentile(0.50));   r.tr_p99   = us(h_tr.percentile(0.99));
    r.tr_p999 = us(h_tr.percentile(0.999));
    r.prod_gaps = prod_gaps; r.cons_gaps = cons_gaps;
    r.worst_gap_ns = prod_worst_gap > cons_worst_gap ? prod_worst_gap : cons_worst_gap;
    r.nivcsw = static_cast<long>(ru1.ru_nivcsw - ru0.ru_nivcsw);
    r.push_retries = push_retries;
    (void)sink;
    return r;
}

// ── Saturation throughput: no rate limiter, the queue is the bottleneck ──────
template <typename Adapter>
double run_throughput(std::uint64_t n, bool bulk) {
    Adapter adapter;
    volatile float sink = 0.0f;
    GameEvent proto = make_payload();
    const std::uint64_t t0 = ticks_now();
    BenchThread consumer([&] {
        std::vector<GameEvent> batch(kBatch);
        std::uint64_t i = 0;
        while (i < n) {
            const std::size_t k = adapter.pop_batch(batch.data(), kBatch, bulk);
            for (std::size_t j = 0; j < k && i < n; ++j, ++i) consume_work(batch[j], &sink);
        }
    }, g_pin_consumer);
    BenchThread producer([&] {
        for (std::uint64_t i = 0; i < n; ++i) while (!adapter.push(proto)) {}
        adapter.stop();
    }, g_pin_producer);
    producer.join(); consumer.join();
    const double secs = static_cast<double>(ticks_to_ns(
        static_cast<std::int64_t>(ticks_now() - t0))) / 1e9;
    (void)sink;
    return static_cast<double>(n) / secs / 1e6;   // M msg/s
}

// ── Payload-size sweep ───────────────────────────────────────────────────────
//
// Answers the question an interviewer will ask about a ns/op figure: is this
// bound by cache-coherence traffic on the index lines, or by copying the
// message? If ns/op is flat across 16 -> 112 bytes the cost is coherence and
// the payload is free; if it scales with size the copy dominates, and that is
// what the zero-copy claim/commit path removes.
template <std::size_t N>
struct Payload {
    std::int64_t stamp;
    char         filler[N - sizeof(std::int64_t)];
};
static_assert(sizeof(Payload<16>) == 16, "payload must be exactly its nominal size");
static_assert(sizeof(Payload<112>) == 112, "payload must be exactly its nominal size");

template <typename T>
inline void touch(const T& v, volatile std::int64_t* sink) noexcept { *sink = v.stamp; }

enum class Path { Single, PushBulk, PopBulk, ZeroCopy };

// Saturated throughput for one message type and one API path. Returns M msg/s.
template <typename T>
double throughput_path(std::uint64_t n, Path path) {
    gridiron::RingBuffer<T, kCapacity> ring;
    volatile std::int64_t sink = 0;
    T proto{};
    proto.stamp = 1;

    const std::uint64_t t0 = ticks_now();

    BenchThread consumer([&] {
        std::vector<T> batch(kBatch);
        std::uint64_t i = 0;
        while (i < n) {
            if (path == Path::PopBulk) {
                const std::size_t k = ring.pop_bulk(batch.data(), kBatch);
                for (std::size_t j = 0; j < k && i < n; ++j, ++i) touch(batch[j], &sink);
            } else if (path == Path::ZeroCopy) {
                // Read straight out of the slot; no copy into a local at all.
                const T* ev = ring.peek();
                if (ev == nullptr) continue;
                touch(*ev, &sink);
                ring.consume();
                ++i;
            } else {
                T ev{};
                if (!ring.pop(ev)) continue;
                touch(ev, &sink);
                ++i;
            }
        }
    }, g_pin_consumer);

    BenchThread producer([&] {
        if (path == Path::PushBulk) {
            std::vector<T> out(kBatch, proto);
            std::uint64_t i = 0;
            while (i < n) {
                const std::size_t want =
                    static_cast<std::size_t>(n - i) < kBatch ? static_cast<std::size_t>(n - i) : kBatch;
                i += ring.push_bulk(out.data(), want);
            }
        } else if (path == Path::ZeroCopy) {
            for (std::uint64_t i = 0; i < n; ++i) {
                T* slot = nullptr;
                while ((slot = ring.claim()) == nullptr) {}
                slot->stamp = static_cast<std::int64_t>(i);   // construct in place
                ring.commit();
            }
        } else {
            for (std::uint64_t i = 0; i < n; ++i) while (!ring.push(proto)) {}
        }
    }, g_pin_producer);

    producer.join();
    consumer.join();
    const double secs = static_cast<double>(
        ticks_to_ns(static_cast<std::int64_t>(ticks_now() - t0))) / 1e9;
    (void)sink;
    return static_cast<double>(n) / secs / 1e6;
}

const char* path_name(Path p) {
    switch (p) {
        case Path::Single:   return "push/pop (single)";
        case Path::PushBulk: return "push_bulk(32) + pop (single)";
        case Path::PopBulk:  return "push (single) + pop_bulk(32)";
        case Path::ZeroCopy: return "claim/commit + peek/consume";
    }
    return "?";
}

// ── CLI ──────────────────────────────────────────────────────────────────────
struct Options {
    std::vector<std::uint64_t> rates{100000, 500000, 1000000};
    double warmup_s = 2.0, duration_s = 10.0, cooldown_s = 2.0;
    int    reps = 5;
    std::uint64_t thr_msgs = 20000000;
    std::string mode = "both";
    std::vector<std::size_t> payloads{16, 64, 112};
};

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* nm) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", nm); std::exit(2); }
            return argv[++i];
        };
        if      (a == "--warmup-s")   o.warmup_s   = std::stod(need("--warmup-s"));
        else if (a == "--duration-s") o.duration_s = std::stod(need("--duration-s"));
        else if (a == "--cooldown-s") o.cooldown_s = std::stod(need("--cooldown-s"));
        else if (a == "--reps")       o.reps       = std::stoi(need("--reps"));
        else if (a == "--mode")       o.mode       = need("--mode");
        else if (a == "--throughput-msgs") o.thr_msgs = std::stoull(need("--throughput-msgs"));
        else if (a == "--payloads") {
            o.payloads.clear();
            std::stringstream ps(need("--payloads")); std::string tk;
            while (std::getline(ps, tk, ',')) o.payloads.push_back(std::stoull(tk));
        }
        else if (a == "--pin-producer") g_pin_producer = std::stoi(need("--pin-producer"));
        else if (a == "--pin-consumer") g_pin_consumer = std::stoi(need("--pin-consumer"));
        else if (a == "--rates") {
            o.rates.clear();
            std::stringstream ss(need("--rates")); std::string tok;
            while (std::getline(ss, tok, ',')) o.rates.push_back(std::stoull(tok));
        } else if (a == "-h" || a == "--help") {
            std::printf(
                "spsc_latency_bench -- open-loop latency + saturation throughput\n"
                "  --mode latency|throughput|both   (default both)\n"
                "  --rates 100000,500000,1000000\n"
                "  --warmup-s 2  --duration-s 10  --cooldown-s 2\n"
                "  --reps 5                         median + [min-max] per cell\n"
                "  --throughput-msgs 20000000\n"
                "  --mode payload                   ns/op at 16/64/112 B x API path\n"
                "  --payloads 16,64,112             sizes for the payload sweep\n"
                "  --pin-producer N --pin-consumer M   Linux only: hard core pinning.\n"
                "                                   Pick isolated cores, ideally\n"
                "                                   siblings-free and on one NUMA node:\n"
                "                                     isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3\n"
                "                                   then --pin-producer 2 --pin-consumer 3\n");
            std::exit(0);
        } else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); std::exit(2); }
    }
    return o;
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

struct Cell { std::uint64_t rate; std::string impl; std::vector<LatResult> reps; };

void cool(double s) {
    if (s > 0) std::this_thread::sleep_for(std::chrono::duration<double>(s));
}

}  // namespace

int main(int argc, char** argv) {
    init_timebase();
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    const Options opt = parse_args(argc, argv);

    std::printf("# Gridiron SPSC bench -- open-loop latency + saturation throughput\n");
    std::printf("# compiler: %s  C++17  -O3 -DNDEBUG -march=native\n",
#if defined(__clang__)
        "clang " __clang_version__
#else
        "unknown"
#endif
    );
    std::printf("# sizeof(GameEvent)=%zu  ring capacity=%zu (%zu live)  batch=%zu\n",
                sizeof(GameEvent), kCapacity, kQueueCap, kBatch);
    std::printf("# clock: %s, quantum %.2f ns -- latencies below ~%.0f ns are at\n"
                "#        timer resolution, do not read them to the nanosecond\n",
                clock_name(), g_ns_per_tick, g_ns_per_tick * 3);
#if defined(__linux__)
    if (g_pin_producer >= 0 || g_pin_consumer >= 0) {
        std::printf("# placement: producer pinned to CPU %d, consumer to CPU %d\n",
                    g_pin_producer, g_pin_consumer);
    } else {
        std::printf("# placement: UNPINNED. Pass --pin-producer/--pin-consumer with\n"
                    "#            isolated cores, or the tail measures the scheduler.\n");
    }
#elif defined(__APPLE__)
    std::printf("# placement: macOS cannot pin threads; QOS_CLASS_USER_INTERACTIVE is a\n"
                "#            P-core bias only. Expect run-to-run spread from cluster\n"
                "#            placement that repetition cannot remove.\n");
#endif
    std::printf("# warmup %.0fs, measure %.0fs, cooldown %.0fs, reps=%d (interleaved)\n\n",
                opt.warmup_s, opt.duration_s, opt.cooldown_s, opt.reps);

    // ── Saturation throughput ────────────────────────────────────────────────
    if (opt.mode == "both" || opt.mode == "throughput") {
        std::printf("## Saturation throughput (no rate limit; the queue is the bottleneck)\n");
        std::printf("%-30s %14s %12s %16s\n", "impl", "median Mmsg/s", "ns/op", "[min-max]");
        enum Kind { kLegacy, kSpsc, kMutex };
        struct TCell { const char* nm; Kind kind; bool bulk; std::vector<double> v; };
        std::vector<TCell> t = {
            {"SPSC, pre-fix (acquire/op)",  kLegacy, false, {}},
            {"lock-free SPSC",              kSpsc,   false, {}},
            {"lock-free SPSC (bulk 32)",    kSpsc,   true,  {}},
            {"mutex + std::queue",          kMutex,  false, {}},
            {"mutex + std::queue (bulk 32)",kMutex,  true,  {}},
        };
        for (int r = 0; r < opt.reps; ++r) {
            for (auto& c : t) {   // interleaved: every rep touches every variant
                cool(opt.cooldown_s);
                const double v =
                    c.kind == kLegacy ? run_throughput<LegacySpscAdapter>(opt.thr_msgs, false)
                  : c.kind == kSpsc   ? run_throughput<SpscAdapter>(opt.thr_msgs, c.bulk)
                                      : run_throughput<MutexAdapter>(opt.thr_msgs, c.bulk);
                c.v.push_back(v);
            }
        }
        for (auto& c : t) {
            auto s = c.v; std::sort(s.begin(), s.end());
            const double med = median_of(c.v);
            std::printf("%-30s %14.2f %12.1f   [%.1f-%.1f]\n",
                        c.nm, med, 1e3 / med, s.front(), s.back());
        }
        std::printf("\n");
    }

    // ── Payload-size / API-path sweep ────────────────────────────────────────
    if (opt.mode == "payload") {
        std::printf("## Throughput by payload size and API path (median of %d reps, "
                    "%llu msgs/rep)\n", opt.reps, (unsigned long long)opt.thr_msgs);
        std::printf("%-30s %10s %14s %12s\n", "path", "payload B", "median Mmsg/s", "ns/op");
        const Path paths[] = {Path::Single, Path::PushBulk, Path::PopBulk, Path::ZeroCopy};
        for (Path path : paths) {
            for (std::size_t bytes : opt.payloads) {
                std::vector<double> v;
                for (int r = 0; r < opt.reps; ++r) {
                    cool(opt.cooldown_s);
                    // Sizes are compile-time; a small dispatch keeps one code path.
                    const double x =
                        bytes <= 16  ? throughput_path<Payload<16>>(opt.thr_msgs, path)
                      : bytes <= 32  ? throughput_path<Payload<32>>(opt.thr_msgs, path)
                      : bytes <= 64  ? throughput_path<Payload<64>>(opt.thr_msgs, path)
                      : bytes <= 112 ? throughput_path<Payload<112>>(opt.thr_msgs, path)
                                     : throughput_path<Payload<256>>(opt.thr_msgs, path);
                    v.push_back(x);
                }
                const double med = median_of(v);
                std::printf("%-30s %10zu %14.2f %12.1f\n",
                            path_name(path), bytes, med, 1e3 / med);
            }
        }
        std::printf("\n# Flat ns/op across payload sizes => bound by index-line coherence.\n"
                    "# ns/op rising with size => bound by the message copy, which is what\n"
                    "# the claim/commit + peek/consume row removes.\n\n");
    }

    // ── Open-loop latency ────────────────────────────────────────────────────
    if (opt.mode == "both" || opt.mode == "latency") {
        std::vector<Cell> cells;
        for (std::uint64_t rate : opt.rates) {
            cells.push_back({rate, "SPSC, pre-fix (acquire/op)", {}});
            cells.push_back({rate, "lock-free SPSC", {}});
            cells.push_back({rate, "lock-free SPSC (bulk 32)", {}});
            cells.push_back({rate, "mutex + std::queue", {}});
            cells.push_back({rate, "mutex + std::queue (bulk 32)", {}});
        }
        for (int r = 0; r < opt.reps; ++r) {
            for (auto& c : cells) {
                cool(opt.cooldown_s);
                std::fprintf(stderr, "# rep %d/%d  %llu msg/s  %s\n", r + 1, opt.reps,
                             (unsigned long long)c.rate, c.impl.c_str());
                const bool bulk   = c.impl.find("bulk")    != std::string::npos;
                const bool legacy = c.impl.find("pre-fix") != std::string::npos;
                const bool spsc   = c.impl.find("SPSC")    != std::string::npos;
                c.reps.push_back(
                    legacy ? run_latency<LegacySpscAdapter>(c.rate, opt.warmup_s, opt.duration_s, false)
                  : spsc   ? run_latency<SpscAdapter>(c.rate, opt.warmup_s, opt.duration_s, bulk)
                           : run_latency<MutexAdapter>(c.rate, opt.warmup_s, opt.duration_s, bulk));
            }
        }

        std::printf("## End-to-end latency, us (median of %d reps; latency measured from "
                    "intended send time)\n", opt.reps);
        std::printf("%-12s %-30s %9s %9s %9s %9s\n",
                    "Rate", "Impl", "p50", "p99", "p99.9", "max");
        for (auto& c : cells) {
            auto pick = [&](double LatResult::*f) {
                std::vector<double> v; for (auto& r : c.reps) v.push_back(r.*f);
                return median_of(v);
            };
            std::printf("%-12llu %-30s %9.2f %9.2f %9.2f %9.2f\n",
                        (unsigned long long)c.rate, c.impl.c_str(),
                        pick(&LatResult::e2e_p50), pick(&LatResult::e2e_p99),
                        pick(&LatResult::e2e_p999), pick(&LatResult::e2e_max));
        }

        std::printf("\n## Where the latency comes from, us (median of %d reps)\n", opt.reps);
        std::printf("%-12s %-30s %19s %19s %14s\n",
                    "Rate", "Impl", "prod lateness p99", "transport p99", "e2e p99");
        for (auto& c : cells) {
            auto pick = [&](double LatResult::*f) {
                std::vector<double> v; for (auto& r : c.reps) v.push_back(r.*f);
                return median_of(v);
            };
            std::printf("%-12llu %-30s %19.2f %19.2f %14.2f\n",
                        (unsigned long long)c.rate, c.impl.c_str(),
                        pick(&LatResult::late_p99), pick(&LatResult::tr_p99),
                        pick(&LatResult::e2e_p99));
        }

        std::printf("\n## Scheduler interference (median of %d reps)\n", opt.reps);
        std::printf("%-12s %-30s %12s %12s %14s %12s\n",
                    "Rate", "Impl", "prod gaps", "cons gaps", "worst gap us", "nivcsw");
        for (auto& c : cells) {
            auto pickd = [&](std::uint64_t LatResult::*f) {
                std::vector<double> v; for (auto& r : c.reps) v.push_back((double)(r.*f));
                return median_of(v);
            };
            std::vector<double> wg, ni;
            for (auto& r : c.reps) { wg.push_back(r.worst_gap_ns / 1000.0); ni.push_back((double)r.nivcsw); }
            std::printf("%-12llu %-30s %12.0f %12.0f %14.1f %12.0f\n",
                        (unsigned long long)c.rate, c.impl.c_str(),
                        pickd(&LatResult::prod_gaps), pickd(&LatResult::cons_gaps),
                        median_of(wg), median_of(ni));
        }
        std::printf("\n# 'prod gaps'/'cons gaps' = times that thread's own clock jumped >%lldus\n"
                    "# between consecutive reads, i.e. it lost the core. Blocking impls report\n"
                    "# 0 consumer gaps by construction -- they sleep instead of spinning.\n",
                    (long long)(kGapThresholdNs / 1000));
    }

    if (!g_pin_ok)
        std::printf("\n# WARNING: core pinning was requested but pthread_setaffinity_np "
                    "failed; these numbers are unpinned.\n");
    if (!g_qos_ok)
        std::printf("\n# WARNING: a thread did not get QOS_CLASS_USER_INTERACTIVE; "
                    "E-core placement may distort these numbers.\n");
    return 0;
}
