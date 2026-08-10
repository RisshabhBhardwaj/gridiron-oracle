#!/usr/bin/env bash
# Shared HTTP assertion helper for e2e/smoke_test.sh.
#
# Extracted so it can be tested on its own (backend/tests/test_http_assert.py).
# `bash -n` passing is not a test, and neither is `grep -qE '\[|\]'` — that
# passes on an empty array and on most error payloads (audit C-32).
#
# Requires: PYTHON, and PASS / FAIL counters defined by the caller.

: "${PYTHON:=python3}"
: "${PASS:=0}"
: "${FAIL:=0}"

RESP_BODY="${RESP_BODY:-$(mktemp -t gridiron_smoke)}"

http_assert_cleanup() {
  [ -n "${RESP_BODY:-}" ] && rm -f "$RESP_BODY"
}

# api_check <label> <url> <python-assertions>
#
# Asserts HTTP 200, then validates the response body's SHAPE. The assertions
# run in $PYTHON with the decoded body bound to `d` and `json` imported; `jq`
# is not assumed to be installed.
api_check() {
  local label="$1" url="$2" assertions="$3"
  local status why

  status="$(curl -sS -o "$RESP_BODY" -w '%{http_code}' "$url" 2>/dev/null || echo 000)"
  if [ "$status" != "200" ]; then
    echo "  FAIL  $label (HTTP $status)"
    head -c 300 "$RESP_BODY" 2>/dev/null | sed 's/^/        /'
    echo ""
    FAIL=$((FAIL + 1))
    return 1
  fi

  if why="$($PYTHON -c "
import json
d = json.load(open('$RESP_BODY'))
$assertions
" 2>&1)"; then
    echo "  PASS  $label"
    PASS=$((PASS + 1))
    return 0
  fi

  echo "  FAIL  $label (HTTP 200, schema assertion failed)"
  printf '%s\n' "$why" | tail -3 | sed 's/^/        /'
  FAIL=$((FAIL + 1))
  return 1
}

# status_check <label> <url> <expected-status>
#
# For endpoints whose correct answer is not 200 — e.g. an obsolete request
# shape that must be rejected.
status_check() {
  local label="$1" url="$2" expected="$3"
  local status
  status="$(curl -sS -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo 000)"
  if [ "$status" = "$expected" ]; then
    echo "  PASS  $label (HTTP $status)"
    PASS=$((PASS + 1))
    return 0
  fi
  echo "  FAIL  $label (HTTP $status, expected $expected)"
  FAIL=$((FAIL + 1))
  return 1
}
