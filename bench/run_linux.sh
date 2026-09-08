#!/usr/bin/env bash
# bench/run_linux.sh — run the SPSC bench on Linux with pinned, isolated cores.
#
# Why this script exists: a latency tail measured on a shared, frequency-scaling,
# SMT-enabled core is a measurement of the scheduler and the governor, not of the
# queue. This captures the environment alongside the numbers so a reader can tell
# which one they are looking at.
#
# Usage:
#   ./bench/run_linux.sh                 # auto-pick two cores, warn if not isolated
#   ./bench/run_linux.sh 2 3             # pin producer to CPU 2, consumer to CPU 3
#   MODE=payload ./bench/run_linux.sh 2 3
#
# For numbers worth quoting, boot with two cores fully handed over to you:
#   isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3
# and disable the SMT sibling of each chosen core (see the SMT section below).
# No sudo is required to *run* this; the tuning steps that need root are printed,
# not executed.

set -u

PROD_CPU="${1:-}"
CONS_CPU="${2:-}"
MODE="${MODE:-both}"
REPS="${REPS:-9}"
MSGS="${MSGS:-20000000}"
RATES="${RATES:-100000,500000,1000000}"
BUILD_DIR="${BUILD_DIR:-engine/build}"
OUT="${OUT:-bench-results-$(date +%Y%m%d-%H%M%S).txt}"

if [ "$(uname -s)" != "Linux" ]; then
  echo "This script is Linux-only (it pins cores). On macOS use: make latency-bench" >&2
  exit 1
fi

# ── Pick cores ───────────────────────────────────────────────────────────────
ISOLATED=""
if [ -r /sys/devices/system/cpu/isolated ]; then
  ISOLATED="$(cat /sys/devices/system/cpu/isolated 2>/dev/null || true)"
fi

if [ -z "$PROD_CPU" ] || [ -z "$CONS_CPU" ]; then
  if [ -n "$ISOLATED" ]; then
    # Take the first two isolated CPUs, expanding a leading "a-b" range.
    FIRST_TOK="$(echo "$ISOLATED" | cut -d, -f1)"
    case "$FIRST_TOK" in
      *-*) PROD_CPU="${FIRST_TOK%%-*}"; CONS_CPU="$((PROD_CPU + 1))" ;;
      *)   PROD_CPU="$FIRST_TOK"; CONS_CPU="$(echo "$ISOLATED" | cut -d, -f2)" ;;
    esac
  fi
  # Fall back to the last two CPUs, which are least likely to hold IRQs.
  if [ -z "${PROD_CPU:-}" ] || [ -z "${CONS_CPU:-}" ]; then
    NCPU="$(getconf _NPROCESSORS_ONLN)"
    PROD_CPU="$((NCPU - 2))"
    CONS_CPU="$((NCPU - 1))"
  fi
fi

# ── Environment report (this is part of the result, not decoration) ──────────
{
  echo "# ── run environment ──────────────────────────────────────────────"
  echo "# date:        $(date -Is)"
  echo "# host:        $(uname -srm)"
  echo "# cpu:         $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
  echo "# cores:       $(getconf _NPROCESSORS_ONLN) online"
  echo "# pinned:      producer=CPU${PROD_CPU}  consumer=CPU${CONS_CPU}"
  echo "# isolcpus:    ${ISOLATED:-<none — cores are shared with the scheduler>}"
  echo "# nohz_full:   $(cat /sys/devices/system/cpu/nohz_full 2>/dev/null || echo '<none>')"

  for c in "$PROD_CPU" "$CONS_CPU"; do
    GOV="$(cat /sys/devices/system/cpu/cpu${c}/cpufreq/scaling_governor 2>/dev/null || echo '?')"
    CUR="$(cat /sys/devices/system/cpu/cpu${c}/cpufreq/scaling_cur_freq 2>/dev/null || echo '?')"
    SIB="$(cat /sys/devices/system/cpu/cpu${c}/topology/thread_siblings_list 2>/dev/null || echo '?')"
    echo "# cpu${c}:       governor=${GOV} cur_khz=${CUR} smt_siblings=${SIB}"
  done

  echo "# turbo:       intel_no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo 'n/a') boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 'n/a')"
  echo "# smt:         $(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo 'n/a')"
  echo "#"
} | tee "$OUT"

# ── Warn loudly about anything that will distort the tail ────────────────────
WARN=0
if [ -z "$ISOLATED" ]; then
  echo "# WARNING: no isolcpus. The kernel may schedule other work on CPU${PROD_CPU}/${CONS_CPU}." | tee -a "$OUT"
  echo "#          Boot with: isolcpus=${PROD_CPU},${CONS_CPU} nohz_full=${PROD_CPU},${CONS_CPU} rcu_nocbs=${PROD_CPU},${CONS_CPU}" | tee -a "$OUT"
  WARN=1
fi
for c in "$PROD_CPU" "$CONS_CPU"; do
  GOV="$(cat /sys/devices/system/cpu/cpu${c}/cpufreq/scaling_governor 2>/dev/null || echo '')"
  if [ -n "$GOV" ] && [ "$GOV" != "performance" ]; then
    echo "# WARNING: cpu${c} governor is '${GOV}', not 'performance'. Frequency scaling" | tee -a "$OUT"
    echo "#          will show up as latency. Fix (needs root):" | tee -a "$OUT"
    echo "#            sudo cpupower -c ${c} frequency-set -g performance" | tee -a "$OUT"
    WARN=1
  fi
  SIB="$(cat /sys/devices/system/cpu/cpu${c}/topology/thread_siblings_list 2>/dev/null || echo '')"
  case "$SIB" in
    *,*|*-*)
      echo "# WARNING: cpu${c} shares a physical core with ${SIB}. An SMT sibling steals" | tee -a "$OUT"
      echo "#          execution resources. Offline it, or disable SMT (needs root):" | tee -a "$OUT"
      echo "#            echo off | sudo tee /sys/devices/system/cpu/smt/control" | tee -a "$OUT"
      WARN=1 ;;
  esac
done
if [ "$WARN" -eq 0 ]; then
  echo "# environment looks clean: isolated, pinned, performance governor, no SMT sibling." | tee -a "$OUT"
fi
echo "" | tee -a "$OUT"

# ── Build ────────────────────────────────────────────────────────────────────
cmake -S engine -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" --target spsc_latency_bench -j"$(getconf _NPROCESSORS_ONLN)" >/dev/null
BIN="$BUILD_DIR/spsc_latency_bench"

PIN="--pin-producer $PROD_CPU --pin-consumer $CONS_CPU"

# chrt gives SCHED_FIFO if permitted; without it the threads are still pinned,
# just preemptible by other runnable work on the same core.
RUNNER=""
if command -v chrt >/dev/null 2>&1 && chrt -f 80 true >/dev/null 2>&1; then
  RUNNER="chrt -f 80"
  echo "# using SCHED_FIFO 80 (chrt available)" | tee -a "$OUT"
else
  echo "# note: SCHED_FIFO unavailable (needs CAP_SYS_NICE); running SCHED_OTHER." | tee -a "$OUT"
  echo "#       For the cleanest tail: sudo setcap cap_sys_nice+ep $BIN" | tee -a "$OUT"
fi
echo "" | tee -a "$OUT"

# ── Run ──────────────────────────────────────────────────────────────────────
case "$MODE" in
  throughput|both)
    $RUNNER "$BIN" --mode throughput --reps "$REPS" \
      --throughput-msgs "$MSGS" --cooldown-s 1 $PIN 2>/dev/null | tee -a "$OUT" ;;
esac
case "$MODE" in
  payload)
    $RUNNER "$BIN" --mode payload --reps "$REPS" \
      --throughput-msgs "$MSGS" --cooldown-s 1 $PIN 2>/dev/null | tee -a "$OUT" ;;
esac
case "$MODE" in
  latency|both)
    $RUNNER "$BIN" --mode latency --reps 3 --warmup-s 2 --duration-s 10 \
      --cooldown-s 1 --rates "$RATES" $PIN 2>/dev/null | tee -a "$OUT" ;;
esac

echo "" | tee -a "$OUT"
echo "# results written to $OUT" | tee -a "$OUT"
