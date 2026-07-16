#!/bin/bash
# bench_05_agentic.sh — agentic coding capability benchmark via qwen-code
# Part of the bench_NN_* series. Unlike 01-04 (raw pp/tg throughput), this
# measures whether a model driven by qwen-code (headless, --approval-mode yolo)
# actually SOLVES programming tasks end-to-end. Twelve tasks, increasing
# difficulty, each verified objectively by this script (protected-file
# checksums + our own pytest / functional checks — the agent's own claims
# and self-written tests are never trusted for the verdict). Every verdict
# also carries a SCORE (tests passed / total, percent) — including TIMEOUT,
# which is scored on whatever the agent left behind at the cutoff.
#   bugfix    — failing pytest, find & fix a bug in stats.py
#   scratch   — build a CLI tool + own tests from an empty directory
#   lru       — implement LRUCache (capacity + TTL + injectable clock) from
#               a provided test suite only (TDD-style)
#   multifile — three distinct bugs across three modules (mutable default,
#               truncation vs half-up rounding, state aliasing at checkout)
#   intervals — implement a booking Scheduler (half-open overlap rejection,
#               exact-match cancel, earliest-free-slot search) from tests
#   fsm       — implement an Order state machine (guarded transitions,
#               accumulating payment with atomic overpay rejection,
#               refunds, state history) from tests
#   codec     — implement a binary frame codec (LEB128 varint length +
#               payload + XOR checksum) from tests with exact byte-literal
#               assertions; malformed input must raise. Byte/bit-level
#               reasoning axis — small code, trap-dense, fast to verify
#   toposort  — implement a deterministic topological sort (the
#               lexicographically smallest valid order — requires a heap,
#               naive FIFO Kahn fails) with CycleError(ValueError) cycle
#               detection. Graph + determinism axis, equally quick
#   template  — implement a mini template engine ({{ var }}, dotted lookup,
#               {% if %}/{% else %}, nestable {% for %}) from tests
#   interp    — implement an expression evaluator (precedence, right-assoc ^,
#               variables, short-circuit and/or, lazy conditional, functions)
#               from tests; eval/exec/compile forbidden and grep-verified
#   perf      — implement EventLog with range queries under a hard time
#               budget (400K adds + 100K queries < 10 s: naive scan and
#               insort both fail; needs lazy sort + bisect)
#   regex     — implement a backtracking regex engine (fullmatch with
#               capture groups: classes, ranges, negation, \d \w \s,
#               greedy * + ?, alternation, nested groups) from tests;
#               import re/regex forbidden and AST-verified
# Description: BENCH.md
set -euo pipefail

MODELS="${MODELS:-qwen36-128k,qwythos}"
TASKS="${TASKS:-bugfix,scratch,lru,multifile,intervals,fsm,codec,toposort,template,interp,perf,regex}"
TASK_TIMEOUT="${TASK_TIMEOUT:-900}"
WORKROOT="${WORKROOT:-/tmp/bench-agentic}"
AILLAMA_BIN="${AILLAMA_BIN:-aillama}"
QWEN_BIN="${QWEN_BIN:-qwen}"
export QWEN_CODE_SUPPRESS_YOLO_WARNING=1

die() { echo "ERROR: $*" >&2; exit 1; }

# Run the task's pytest suite (quiet, no tracebacks), write the pass ratio
# "passed/total (pct%)" to $1/.score, and succeed only when every test
# passed. $3 = expected test count of a protected suite, so a missing or
# unimportable solution scores 0/N instead of 0/0 (0 = agent-written suite,
# total is whatever pytest collected). Written to a file, not a global,
# because verify_* runs inside a $(...) subshell in run_task.
scored_pytest() {
	local dir="$1" tmo="${2:-120}" expected="${3:-0}" out passed failed total
	out=$( (cd "$dir" && timeout "$tmo" python -m pytest -q --tb=no 2>&1) || true )
	passed=$(grep -oE '[0-9]+ passed' <<<"$out" | tail -1 | cut -d' ' -f1) || true
	failed=$(grep -oE '[0-9]+ (failed|error)' <<<"$out" | awk '{s+=$1} END {print s+0}') || true
	passed=${passed:-0}; failed=${failed:-0}
	total=$(( passed + failed ))
	(( total < expected )) && total=$expected
	if (( total > 0 )); then
		echo "$passed/$total ($(( 100 * passed / total ))%)" >"$dir/.score"
	fi
	(( passed > 0 && passed == total ))
}

pomoc() {
	cat <<EOF
Usage: ${0##*/} [-h]

Agentic coding benchmark: for each aillama profile in MODELS, switch the
llama-server to it and run each task in TASKS through headless qwen-code,
then verify the result objectively (script-side pytest + checksums).
Prints a summary table: verdict (PASS/FAIL/TIMEOUT), SCORE (tests passed /
total with percentage — TIMEOUT is scored on the work left at the cutoff)
and wall time.

The server is left running on the LAST profile in MODELS.

Environment variables (current values):
  MODELS        aillama profiles to compare   ($MODELS)
  TASKS         task subset                    ($TASKS)
  TASK_TIMEOUT  seconds per task               ($TASK_TIMEOUT)
  WORKROOT      scratch directory root         ($WORKROOT)
  AILLAMA_BIN   aillama binary                 ($AILLAMA_BIN)
  QWEN_BIN      qwen-code binary               ($QWEN_BIN)
EOF
	exit 0
}

# ---------------------------------------------------------------- tasks ----

setup_bugfix() {
	local dir="$1"
	cat >"$dir/stats.py" <<'EOF'
def median(values):
    """Return the median of a list of numbers."""
    s = sorted(values)
    n = len(s)
    return s[n // 2]


def mode(values):
    """Return the most frequent value (first one wins on ties)."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = None
    for v, c in counts.items():
        if best is None or c > counts[best]:
            best = v
    return best
EOF
	cat >"$dir/test_stats.py" <<'EOF'
from stats import median, mode

def test_median_odd():
    assert median([3, 1, 2]) == 2

def test_median_even():
    assert median([4, 1, 3, 2]) == 2.5

def test_median_single():
    assert median([7]) == 7

def test_mode():
    assert mode([1, 2, 2, 3]) == 2
EOF
	sha256sum "$dir/test_stats.py" >"$dir/.protected.sha"
}

prompt_bugfix() {
	echo "Run 'python -m pytest -q' in this directory. One test fails. Find the bug in stats.py, fix it by editing the file, and re-run the tests until all pass. Do not modify test_stats.py."
}

verify_bugfix() {
	local dir="$1"
	(cd "$dir" && sha256sum -c .protected.sha >/dev/null 2>&1) || { echo "protected test file modified"; return 1; }
	scored_pytest "$dir" 120 4 || { echo "pytest still failing"; return 1; }
	return 0
}

setup_scratch() {
	local dir="$1"
	: # empty directory by design
}

prompt_scratch() {
	echo "In this empty directory create wordfreq.py: a CLI tool that reads a text file given as the first argument and prints the N most frequent words (option --top N, default 10), case-insensitive, stripping punctuation, output format 'word count' per line sorted by count descending then alphabetically on ties. Also create test_wordfreq.py with pytest tests covering: case folding, punctuation stripping, the --top limit, and tie-breaking order. Run the tests and fix issues until all pass."
}

verify_scratch() {
	# hidden functional check — the agent's own tests are not trusted
	local dir="$1" out expected
	[[ -f "$dir/wordfreq.py" ]] || { echo "wordfreq.py not created"; return 1; }
	printf 'The cat sat. The CAT ran! A dog, the dog?\n' >"$dir/.check.txt"
	out=$(cd "$dir" && timeout 30 python wordfreq.py .check.txt --top 3 2>/dev/null) || { echo "CLI crashed"; return 1; }
	expected=$(printf 'the 3\ncat 2\ndog 2')
	[[ "$out" == "$expected" ]] || { echo "wrong output: $(echo "$out" | tr '\n' '|')"; return 1; }
	scored_pytest "$dir" 120 || { echo "agent's own tests fail"; return 1; }
	return 0
}

setup_lru() {
	local dir="$1"
	cat >"$dir/test_lru.py" <<'EOF'
# Spec-by-tests: implement lru.py with class
#   LRUCache(capacity, ttl=None, clock=time.monotonic)
# get(key) -> value or None; put(key, value); len(cache) = live entries.
import time
from lru import LRUCache

def test_put_get():
    c = LRUCache(capacity=2)
    c.put("a", 1)
    assert c.get("a") == 1

def test_missing_returns_none():
    c = LRUCache(capacity=2)
    assert c.get("nope") is None

def test_eviction_order():
    c = LRUCache(capacity=2)
    c.put("a", 1); c.put("b", 2); c.put("c", 3)   # evicts a (oldest)
    assert c.get("a") is None
    assert c.get("b") == 2 and c.get("c") == 3

def test_get_refreshes_recency():
    c = LRUCache(capacity=2)
    c.put("a", 1); c.put("b", 2)
    c.get("a")               # a becomes most recent
    c.put("c", 3)            # evicts b, not a
    assert c.get("b") is None
    assert c.get("a") == 1

def test_update_refreshes_recency_and_value():
    c = LRUCache(capacity=2)
    c.put("a", 1); c.put("b", 2)
    c.put("a", 10)           # update refreshes recency
    c.put("c", 3)            # evicts b
    assert c.get("b") is None
    assert c.get("a") == 10

def test_ttl_expiry_with_injected_clock():
    t = [100.0]
    c = LRUCache(capacity=4, ttl=5.0, clock=lambda: t[0])
    c.put("a", 1)
    t[0] = 104.9
    assert c.get("a") == 1
    t[0] = 105.1
    assert c.get("a") is None

def test_len_counts_live_entries_only():
    t = [0.0]
    c = LRUCache(capacity=4, ttl=1.0, clock=lambda: t[0])
    c.put("a", 1); c.put("b", 2)
    assert len(c) == 2
    t[0] = 2.0
    assert len(c) == 0

def test_capacity_one():
    c = LRUCache(capacity=1)
    c.put("a", 1); c.put("b", 2)
    assert c.get("a") is None
    assert c.get("b") == 2
EOF
	sha256sum "$dir/test_lru.py" >"$dir/.protected.sha"
}

prompt_lru() {
	echo "This directory contains test_lru.py, a pytest suite that fully specifies an LRU cache with optional TTL and an injectable clock. Read the tests carefully, then implement lru.py so that 'python -m pytest -q' passes all tests. Do not modify test_lru.py. Re-run the tests and fix your implementation until everything passes."
}

verify_lru() {
	local dir="$1"
	(cd "$dir" && sha256sum -c .protected.sha >/dev/null 2>&1) || { echo "protected test file modified"; return 1; }
	[[ -f "$dir/lru.py" ]] || { echo "lru.py not created"; return 1; }
	scored_pytest "$dir" 120 8 || { echo "pytest failing"; return 1; }
	return 0
}

setup_multifile() {
	local dir="$1"
	cat >"$dir/models.py" <<'EOF'
class Item:
    def __init__(self, name, qty, unit_cents):
        self.name = name
        self.qty = qty
        self.unit_cents = unit_cents


class Order:
    def __init__(self, customer, items=[]):
        self.customer = customer
        self.items = items

    def add(self, name, qty, unit_cents):
        self.items.append(Item(name, qty, unit_cents))
EOF
	cat >"$dir/pricing.py" <<'EOF'
def apply_discount(price_cents, pct):
    """Apply a percentage discount, rounding half up to whole cents."""
    return int(price_cents * (100 - pct) / 100)


def order_total_cents(order):
    return sum(i.qty * i.unit_cents for i in order.items)
EOF
	cat >"$dir/store.py" <<'EOF'
from models import Order


class Store:
    def __init__(self):
        self._carts = {}
        self.orders = []

    def add_to_cart(self, customer, name, qty, unit_cents):
        cart = self._carts.setdefault(customer, Order(customer))
        cart.add(name, qty, unit_cents)

    def checkout(self, customer):
        order = self._carts.get(customer) or Order(customer)
        self.orders.append(order)
        return order

    def cart_items(self, customer):
        cart = self._carts.get(customer)
        return cart.items if cart else []
EOF
	cat >"$dir/test_shop.py" <<'EOF'
from models import Order
from pricing import apply_discount, order_total_cents
from store import Store

def test_orders_do_not_share_items():
    a = Order("A")
    b = Order("B")
    a.add("widget", 2, 500)
    assert b.items == []

def test_discount_rounds_half_up():
    assert apply_discount(999, 15) == 849   # 849.15 -> 849
    assert apply_discount(150, 25) == 113   # 112.5  -> 113 (half up)
    assert apply_discount(100, 0) == 100

def test_order_total():
    o = Order("A")
    o.add("w", 2, 500)
    o.add("g", 1, 250)
    assert order_total_cents(o) == 1250

def test_checkout_isolates_history():
    s = Store()
    s.add_to_cart("c1", "widget", 1, 500)
    order = s.checkout("c1")
    s.add_to_cart("c1", "gadget", 2, 250)
    assert [i.name for i in order.items] == ["widget"]
    assert order_total_cents(order) == 500

def test_checkout_clears_cart():
    s = Store()
    s.add_to_cart("c1", "w", 1, 100)
    s.checkout("c1")
    assert s.cart_items("c1") == []
EOF
	sha256sum "$dir/test_shop.py" >"$dir/.protected.sha"
}

prompt_multifile() {
	echo "This small project (models.py, pricing.py, store.py) has a pytest suite test_shop.py in which several tests fail, caused by three distinct bugs located in three different files. Run 'python -m pytest -q', locate all the bugs, fix them by editing the source files, and re-run the tests until all pass. Do not modify test_shop.py."
}

verify_multifile() {
	local dir="$1"
	(cd "$dir" && sha256sum -c .protected.sha >/dev/null 2>&1) || { echo "protected test file modified"; return 1; }
	scored_pytest "$dir" 120 5 || { echo "pytest still failing"; return 1; }
	return 0
}

setup_intervals() {
	local dir="$1"
	cat >"$dir/test_booking.py" <<'EOF'
# Spec-by-tests: implement booking.py with class Scheduler:
#   book(start, end) -> bool      — reserve half-open [start, end); False if
#                                   it overlaps an existing booking
#   cancel(start, end) -> bool    — remove an exact existing booking
#   bookings() -> list[(s, e)]    — current bookings sorted by start
#   next_free(t, duration) -> int — earliest start >= t of a free slot of
#                                   the given length
import pytest
from booking import Scheduler

def test_book_and_list():
    s = Scheduler()
    assert s.book(10, 20) is True
    assert s.bookings() == [(10, 20)]

def test_bookings_sorted():
    s = Scheduler()
    s.book(30, 40); s.book(10, 20)
    assert s.bookings() == [(10, 20), (30, 40)]

def test_reject_duplicate():
    s = Scheduler()
    s.book(10, 20)
    assert s.book(10, 20) is False

def test_reject_partial_overlap():
    s = Scheduler()
    s.book(10, 20)
    assert s.book(15, 25) is False
    assert s.book(5, 15) is False

def test_reject_contained_and_containing():
    s = Scheduler()
    s.book(10, 20)
    assert s.book(12, 18) is False
    assert s.book(5, 25) is False

def test_adjacent_is_free():
    s = Scheduler()
    s.book(10, 20)
    assert s.book(20, 30) is True
    assert s.book(0, 10) is True

def test_invalid_interval_raises():
    s = Scheduler()
    with pytest.raises(ValueError):
        s.book(20, 10)
    with pytest.raises(ValueError):
        s.book(10, 10)

def test_cancel_exact_only():
    s = Scheduler()
    s.book(10, 20)
    assert s.cancel(12, 18) is False
    assert s.cancel(10, 20) is True
    assert s.bookings() == []
    assert s.cancel(10, 20) is False

def test_rebook_after_cancel():
    s = Scheduler()
    s.book(10, 20)
    s.cancel(10, 20)
    assert s.book(15, 25) is True

def test_next_free_empty_and_before():
    s = Scheduler()
    assert s.next_free(7, 10) == 7
    s.book(30, 40)
    assert s.next_free(0, 10) == 0

def test_next_free_skips_bookings():
    s = Scheduler()
    s.book(10, 20); s.book(30, 40)
    assert s.next_free(12, 5) == 20      # inside a booking -> next gap
    assert s.next_free(12, 15) == 40     # the 20-30 gap is too short
    assert s.next_free(15, 10) == 20     # exact fit in the 20-30 gap

def test_next_free_starts_mid_gap():
    s = Scheduler()
    s.book(10, 20); s.book(40, 50)
    assert s.next_free(25, 5) == 25      # fits in the rest of the gap
    assert s.next_free(25, 20) == 50     # 25-40 is only 15 long
EOF
	sha256sum "$dir/test_booking.py" >"$dir/.protected.sha"
}

prompt_intervals() {
	echo "This directory contains test_booking.py, a pytest suite that fully specifies a Scheduler class managing half-open [start, end) time-slot bookings: book(start, end) rejecting any overlap (adjacent slots are fine, invalid intervals raise ValueError), exact-match cancel(start, end), sorted bookings(), and next_free(t, duration) returning the earliest start >= t of a free slot of the given length. Read the tests carefully, then implement booking.py so that 'python -m pytest -q' passes all tests. Do not modify test_booking.py. Re-run the tests and fix your implementation until everything passes."
}

verify_intervals() {
	local dir="$1"
	(cd "$dir" && sha256sum -c .protected.sha >/dev/null 2>&1) || { echo "protected test file modified"; return 1; }
	[[ -f "$dir/booking.py" ]] || { echo "booking.py not created"; return 1; }
	scored_pytest "$dir" 120 12 || { echo "pytest failing"; return 1; }
	return 0
}

setup_fsm() {
	local dir="$1"
	cat >"$dir/test_order.py" <<'EOF'
# Spec-by-tests: implement order.py with class Order(total_cents) and
# exception InvalidTransition. States: "new" -> pay(amount) accumulating up
# to the total -> "paid" -> ship() -> "shipped" -> deliver() -> "delivered".
# cancel() only from new/paid, returns the refunded amount. A payment that
# would exceed the total raises ValueError and must not be applied at all.
# Illegal events raise InvalidTransition and change nothing. o.history is
# the list of states visited so far (failed events leave no trace).
import pytest
from order import Order, InvalidTransition

def test_initial_state():
    o = Order(total_cents=1000)
    assert o.state == "new"
    assert o.history == ["new"]

def test_full_payment_moves_to_paid():
    o = Order(1000)
    o.pay(1000)
    assert o.state == "paid"

def test_partial_payments_accumulate():
    o = Order(1000)
    o.pay(300)
    assert o.state == "new"
    o.pay(700)
    assert o.state == "paid"

def test_overpayment_rejected_atomically():
    o = Order(1000)
    o.pay(300)
    with pytest.raises(ValueError):
        o.pay(800)          # would exceed the total
    assert o.state == "new"
    o.pay(700)              # the rejected 800 must NOT have been applied
    assert o.state == "paid"

def test_ship_requires_paid():
    o = Order(1000)
    with pytest.raises(InvalidTransition):
        o.ship()
    o.pay(1000)
    o.ship()
    assert o.state == "shipped"

def test_deliver_requires_shipped():
    o = Order(1000)
    o.pay(1000)
    with pytest.raises(InvalidTransition):
        o.deliver()
    o.ship()
    o.deliver()
    assert o.state == "delivered"

def test_pay_after_paid_is_illegal():
    o = Order(1000)
    o.pay(1000)
    with pytest.raises(InvalidTransition):
        o.pay(1)

def test_cancel_from_new_refunds_nothing():
    o = Order(1000)
    assert o.cancel() == 0
    assert o.state == "cancelled"

def test_cancel_refunds_partial_payment():
    o = Order(1000)
    o.pay(400)
    assert o.cancel() == 400

def test_cancel_from_paid_refunds_total():
    o = Order(1000)
    o.pay(1000)
    assert o.cancel() == 1000
    assert o.state == "cancelled"

def test_cancel_after_ship_is_illegal():
    o = Order(1000)
    o.pay(1000)
    o.ship()
    with pytest.raises(InvalidTransition):
        o.cancel()
    assert o.state == "shipped"

def test_no_events_after_cancel():
    o = Order(1000)
    o.cancel()
    with pytest.raises(InvalidTransition):
        o.pay(100)
    with pytest.raises(InvalidTransition):
        o.ship()

def test_history_records_states_not_failures():
    o = Order(1000)
    with pytest.raises(InvalidTransition):
        o.ship()
    o.pay(1000)
    o.ship()
    o.deliver()
    assert o.history == ["new", "paid", "shipped", "delivered"]
EOF
	sha256sum "$dir/test_order.py" >"$dir/.protected.sha"
}

prompt_fsm() {
	echo "This directory contains test_order.py, a pytest suite that fully specifies an Order workflow state machine: states new/paid/shipped/delivered/cancelled, an accumulating pay(amount) that must atomically reject (ValueError, nothing applied) any payment exceeding the total, guarded ship()/deliver()/cancel() transitions that raise InvalidTransition when illegal and change nothing, cancel() returning the refunded amount, and a history list of visited states in which failed events leave no trace. Read the tests carefully, then implement order.py so that 'python -m pytest -q' passes all tests. Do not modify test_order.py. Re-run the tests and fix your implementation until everything passes."
}

verify_fsm() {
	local dir="$1"
	(cd "$dir" && sha256sum -c .protected.sha >/dev/null 2>&1) || { echo "protected test file modified"; return 1; }
	[[ -f "$dir/order.py" ]] || { echo "order.py not created"; return 1; }
	scored_pytest "$dir" 120 13 || { echo "pytest failing"; return 1; }
	return 0
}

setup_codec() {
	local dir="$1"
	cat >"$dir/test_codec.py" <<'EOF'
# Spec-by-tests: implement codec.py with encode(frames) and decode(data).
# Wire format, per frame, frames concatenated:
#   varint length N (LEB128: little-endian 7-bit groups, high bit = "more")
#   N payload bytes
#   1 checksum byte: XOR of all payload bytes (0 for an empty payload)
# decode must raise ValueError on any malformed input: truncated varint,
# truncated payload, missing or wrong checksum, dangling trailing bytes.
import pytest
from codec import encode, decode

def test_empty_stream():
    assert encode([]) == b""
    assert decode(b"") == []

def test_single_short_frame():
    data = encode([b"AB"])
    assert data == b"\x02AB\x03"          # len=2, payload, 0x41^0x42=0x03
    assert decode(data) == [b"AB"]

def test_empty_payload_frame():
    data = encode([b""])
    assert data == b"\x00\x00"            # len=0, checksum of nothing = 0
    assert decode(data) == [b""]

def test_multiple_frames():
    frames = [b"", b"x", b"hello"]
    assert decode(encode(frames)) == frames

def test_varint_two_bytes():
    payload = bytes(200)                  # length 200 needs 2 varint bytes
    data = encode([payload])
    assert data[:2] == b"\xc8\x01"        # 200 -> 0xC8 0x01
    assert data[-1] == 0                  # xor of 200 zero bytes
    assert decode(data) == [payload]

def test_varint_boundary_127_128():
    d127 = encode([bytes(127)])
    assert d127[0] == 0x7F and len(d127) == 1 + 127 + 1
    d128 = encode([bytes(128)])
    assert d128[:2] == b"\x80\x01" and len(d128) == 2 + 128 + 1

def test_checksum_xor():
    data = encode([bytes([0xFF, 0x0F, 0xF0])])
    assert data[-1] == 0x00               # FF^0F^F0 = 00

def test_roundtrip_binary():
    frames = [bytes(range(256)), b"\x00\x80\xff"]
    assert decode(encode(frames)) == frames

def test_bad_checksum_raises():
    corrupted = bytearray(encode([b"AB"]))
    corrupted[-1] ^= 0x01
    with pytest.raises(ValueError):
        decode(bytes(corrupted))

def test_truncated_payload_raises():
    with pytest.raises(ValueError):
        decode(b"\x05AB")                 # promises 5 payload bytes, has 2

def test_truncated_varint_raises():
    with pytest.raises(ValueError):
        decode(b"\x80")                   # continuation bit set, no next byte

def test_trailing_garbage_raises():
    with pytest.raises(ValueError):
        decode(encode([b"AB"]) + b"\x02") # dangling partial frame
EOF
	sha256sum "$dir/test_codec.py" >"$dir/.protected.sha"
}

prompt_codec() {
	echo "This directory contains test_codec.py, a pytest suite that fully specifies a small binary frame codec: implement codec.py with encode(frames) taking a list of bytes objects and decode(data) returning that list back. Each frame on the wire is: the payload length as a LEB128 varint (little-endian 7-bit groups, high bit set on all but the last byte), the payload bytes, then one checksum byte equal to the XOR of all payload bytes (0 for an empty payload); frames are simply concatenated. decode must raise ValueError on any malformed input (truncated varint, truncated payload, wrong checksum, dangling trailing bytes). Read the tests carefully — several assert exact byte sequences. Do not modify test_codec.py. Run 'python -m pytest -q' and fix your implementation until all tests pass."
}

verify_codec() {
	local dir="$1"
	(cd "$dir" && sha256sum -c .protected.sha >/dev/null 2>&1) || { echo "protected test file modified"; return 1; }
	[[ -f "$dir/codec.py" ]] || { echo "codec.py not created"; return 1; }
	scored_pytest "$dir" 120 12 || { echo "pytest failing"; return 1; }
	return 0
}

setup_toposort() {
	local dir="$1"
	cat >"$dir/test_toposort.py" <<'EOF'
# Spec-by-tests: implement toposort.py with
#   class CycleError(ValueError)
#   toposort(nodes, edges) -> list
# edges are (u, v) pairs meaning u must come BEFORE v. Among all valid
# orders return the LEXICOGRAPHICALLY SMALLEST one (compare node values).
# Nodes with no edges still appear in the result. An edge naming a node
# not in `nodes` raises ValueError; any cycle raises CycleError.
import pytest
from toposort import toposort, CycleError

def test_linear_chain():
    assert toposort(["a", "b", "c"], [("a", "b"), ("b", "c")]) == ["a", "b", "c"]

def test_lexicographically_smallest():
    nodes = ["a", "c", "b", "d"]
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    assert toposort(nodes, edges) == ["a", "b", "c", "d"]

def test_smallest_globally_not_greedy_by_depth():
    # after "a" both "b" (isolated) and "z" become candidates; "b" wins
    assert toposort(["z", "b", "a"], [("a", "z")]) == ["a", "b", "z"]

def test_isolated_nodes_included_and_sorted():
    assert toposort(["b", "a"], []) == ["a", "b"]

def test_duplicate_edges_are_harmless():
    assert toposort(["a", "b"], [("a", "b"), ("a", "b")]) == ["a", "b"]

def test_larger_deterministic():
    nodes = ["t3", "t1", "t2", "s", "u"]
    edges = [("s", "t1"), ("s", "t2"), ("s", "t3"),
             ("t1", "u"), ("t2", "u"), ("t3", "u")]
    assert toposort(nodes, edges) == ["s", "t1", "t2", "t3", "u"]

def test_self_loop_is_cycle():
    with pytest.raises(CycleError):
        toposort(["a"], [("a", "a")])

def test_two_node_cycle():
    with pytest.raises(CycleError):
        toposort(["a", "b"], [("a", "b"), ("b", "a")])

def test_cycle_with_acyclic_tail_still_raises():
    with pytest.raises(CycleError):
        toposort(["a", "b", "d"], [("a", "b"), ("b", "a"), ("b", "d")])

def test_cycle_error_is_a_value_error():
    assert issubclass(CycleError, ValueError)

def test_unknown_node_in_edge_raises():
    with pytest.raises(ValueError):
        toposort(["a"], [("a", "ghost")])
EOF
	sha256sum "$dir/test_toposort.py" >"$dir/.protected.sha"
}

prompt_toposort() {
	echo "This directory contains test_toposort.py, a pytest suite that fully specifies a deterministic topological sort: implement toposort.py with a function toposort(nodes, edges) and an exception class CycleError deriving from ValueError. Edges are (u, v) pairs meaning u must come before v. Among all valid topological orders you must return the lexicographically smallest one; think carefully about what data structure makes the smallest ready node come out first. Isolated nodes appear in the result too; an edge naming an unknown node raises ValueError; any cycle (including a self-loop) raises CycleError. Do not modify test_toposort.py. Run 'python -m pytest -q' and fix your implementation until all tests pass."
}

verify_toposort() {
	local dir="$1"
	(cd "$dir" && sha256sum -c .protected.sha >/dev/null 2>&1) || { echo "protected test file modified"; return 1; }
	[[ -f "$dir/toposort.py" ]] || { echo "toposort.py not created"; return 1; }
	scored_pytest "$dir" 120 11 || { echo "pytest failing"; return 1; }
	return 0
}

setup_template() {
	local dir="$1"
	cat >"$dir/test_template.py" <<'EOF'
# Spec-by-tests: implement template.py with render(template_string, context).
from template import render

def test_plain_text():
    assert render("hello", {}) == "hello"

def test_variable():
    assert render("hi {{ name }}!", {"name": "Ala"}) == "hi Ala!"

def test_missing_variable_is_empty():
    assert render("[{{ nope }}]", {}) == "[]"

def test_dotted_lookup():
    assert render("{{ user.name }}", {"user": {"name": "Ola"}}) == "Ola"

def test_if_true_false():
    t = "{% if admin %}yes{% else %}no{% endif %}"
    assert render(t, {"admin": True}) == "yes"
    assert render(t, {"admin": 0}) == "no"

def test_if_without_else():
    assert render("a{% if x %}b{% endif %}c", {"x": False}) == "ac"

def test_for_loop():
    t = "{% for x in items %}{{ x }},{% endfor %}"
    assert render(t, {"items": [1, 2, 3]}) == "1,2,3,"

def test_for_scoping_restores_outer():
    t = "{% for x in items %}{{ x }}{% endfor %}{{ x }}"
    assert render(t, {"items": ["a"], "x": "outer"}) == "aouter"

def test_nested_for_if():
    t = "{% for n in nums %}{% if n %}<{{ n }}>{% endif %}{% endfor %}"
    assert render(t, {"nums": [0, 1, 2]}) == "<1><2>"

def test_nested_loops():
    t = "{% for row in grid %}{% for c in row %}{{ c }}{% endfor %};{% endfor %}"
    assert render(t, {"grid": [[1, 2], [3]]}) == "12;3;"
EOF
	sha256sum "$dir/test_template.py" >"$dir/.protected.sha"
}

prompt_template() {
	echo "This directory contains test_template.py, a pytest suite that fully specifies a minimal text template engine: {{ var }} substitution with dotted lookup, {% if %}/{% else %}/{% endif %} conditionals, and nestable {% for x in seq %} loops with proper variable scoping. Read the tests carefully, then implement template.py with a function render(template_string, context) so that 'python -m pytest -q' passes all tests. Do not modify test_template.py. Re-run the tests and fix your implementation until everything passes."
}

verify_template() {
	local dir="$1"
	(cd "$dir" && sha256sum -c .protected.sha >/dev/null 2>&1) || { echo "protected test file modified"; return 1; }
	[[ -f "$dir/template.py" ]] || { echo "template.py not created"; return 1; }
	scored_pytest "$dir" 120 10 || { echo "pytest failing"; return 1; }
	return 0
}

setup_interp() {
	local dir="$1"
	cat >"$dir/test_calc.py" <<'EOF'
# Spec-by-tests: implement calc.py with evaluate(expr, env=None).
# Semantics are Python-LIKE except: ^ is the power operator (right-assoc).
# eval/exec/compile on the input are forbidden — write a real parser.
import pytest
from calc import evaluate

def test_precedence():
    assert evaluate("2+3*4") == 14

def test_parens():
    assert evaluate("(2+3)*4") == 20

def test_float_division():
    assert evaluate("7/2") == 3.5

def test_unary_minus():
    assert evaluate("-3+5") == 2
    assert evaluate("2*-3") == -6

def test_power_right_assoc():
    assert evaluate("2^3^2") == 512

def test_power_binds_tighter_than_unary():
    assert evaluate("-2^2") == -4

def test_variables():
    assert evaluate("x*2+1", {"x": 10}) == 21

def test_missing_variable_raises():
    with pytest.raises(NameError):
        evaluate("y+1")

def test_comparison():
    assert evaluate("1+2 == 3") is True
    assert evaluate("2 < 1") is False
    assert evaluate("3 >= 3") is True
    assert evaluate("1 != 2") is True

def test_short_circuit_and_or():
    assert evaluate("2 < 1 and 1/0 == 0") is False
    assert evaluate("1 < 2 or 1/0 == 0") is True

def test_division_by_zero_raises_when_evaluated():
    with pytest.raises(ZeroDivisionError):
        evaluate("1/0")

def test_conditional_expression_is_lazy():
    assert evaluate("1 if 2 > 1 else 0") == 1
    assert evaluate("5 if 1 < 2 else 1/0") == 5
    assert evaluate("1/0 if 1 > 2 else 7") == 7

def test_functions():
    assert evaluate("max(1, 2*3)") == 6
    assert evaluate("min(max(1, 2), 10)") == 2
    assert evaluate("abs(-7)") == 7
EOF
	sha256sum "$dir/test_calc.py" >"$dir/.protected.sha"
}

prompt_interp() {
	echo "This directory contains test_calc.py, a pytest suite that fully specifies a small expression evaluator: arithmetic with standard precedence, parentheses, unary minus, a right-associative '^' power operator (NOT xor), variables from an env dict, comparison operators, short-circuit 'and'/'or', a lazy 'X if COND else Y' conditional, and the functions min/max/abs. Implement calc.py with a function evaluate(expr, env=None). You must write the tokenizer/parser/evaluator yourself: calling Python's eval, exec or compile on the input is forbidden and will fail verification. Do not modify test_calc.py. Run 'python -m pytest -q' and fix your implementation until all tests pass."
}

verify_interp() {
	local dir="$1"
	(cd "$dir" && sha256sum -c .protected.sha >/dev/null 2>&1) || { echo "protected test file modified"; return 1; }
	[[ -f "$dir/calc.py" ]] || { echo "calc.py not created"; return 1; }
	# detect BARE builtin eval/exec/compile calls only — a grep would false-
	# positive on re.compile() tokenizers and methods named eval() (seen live)
	local rc=0
	(cd "$dir" && python - <<-'PY'
		import ast, sys
		try:
		    tree = ast.parse(open("calc.py").read())
		except SyntaxError as e:
		    print(f"calc.py does not parse: {e}", file=sys.stderr)
		    sys.exit(2)
		bad = [n.func.id for n in ast.walk(tree)
		       if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
		       and n.func.id in ("eval", "exec", "compile")]
		sys.exit(1 if bad else 0)
	PY
	) || rc=$?
	(( rc == 2 )) && { echo "calc.py has invalid syntax"; return 1; }
	(( rc != 0 )) && { echo "used eval/exec/compile"; return 1; }
	scored_pytest "$dir" 120 13 || { echo "pytest failing"; return 1; }
	return 0
}

setup_perf() {
	local dir="$1"
	cat >"$dir/test_eventlog.py" <<'EOF'
# Spec-by-tests: implement eventlog.py with class EventLog:
#   add(timestamp: int)          — record an event
#   count(start, end) -> int     — events with start <= t < end (half-open)
# The large test enforces a hard time budget: a naive per-query scan or
# per-add insort will NOT finish in time.
import random
import time
from eventlog import EventLog

def test_basic():
    log = EventLog()
    for t in [5, 1, 3]:
        log.add(t)
    assert log.count(1, 4) == 2
    assert log.count(0, 10) == 3
    log.add(2)
    assert log.count(1, 4) == 3

def test_empty():
    log = EventLog()
    assert log.count(0, 100) == 0

def test_half_open_bounds():
    log = EventLog()
    log.add(10)
    assert log.count(10, 11) == 1
    assert log.count(9, 10) == 0
    assert log.count(10, 10) == 0

def test_duplicates():
    log = EventLog()
    for _ in range(3):
        log.add(7)
    assert log.count(7, 8) == 3

def test_correctness_against_bruteforce():
    rng = random.Random(7)
    log = EventLog()
    events = []
    for _ in range(500):
        t = rng.randint(0, 100)
        log.add(t)
        events.append(t)
        a = rng.randint(0, 100)
        b = rng.randint(a, 101)
        assert log.count(a, b) == sum(1 for e in events if a <= e < b)

def test_performance_large():
    rng = random.Random(42)
    log = EventLog()
    start = time.monotonic()
    for _ in range(200_000):
        log.add(rng.randint(0, 10_000_000))
    for _ in range(50_000):
        a = rng.randint(0, 10_000_000)
        log.count(a, a + rng.randint(0, 100_000))
    for _ in range(200_000):
        log.add(rng.randint(0, 10_000_000))
    for _ in range(50_000):
        a = rng.randint(0, 10_000_000)
        log.count(a, a + rng.randint(0, 100_000))
    assert time.monotonic() - start < 10.0
EOF
	sha256sum "$dir/test_eventlog.py" >"$dir/.protected.sha"
}

prompt_perf() {
	echo "This directory contains test_eventlog.py, a pytest suite specifying an EventLog class with add(timestamp) and count(start, end) returning the number of recorded events in the half-open range [start, end). One test performs 400,000 adds and 100,000 range queries under a hard 10-second budget, so both a naive per-query scan and per-add sorted insertion are too slow — think about the data structure and amortized costs. Implement eventlog.py. Do not modify test_eventlog.py. Run 'python -m pytest -q' and fix your implementation until all tests pass within the time budget."
}

verify_perf() {
	local dir="$1"
	(cd "$dir" && sha256sum -c .protected.sha >/dev/null 2>&1) || { echo "protected test file modified"; return 1; }
	[[ -f "$dir/eventlog.py" ]] || { echo "eventlog.py not created"; return 1; }
	scored_pytest "$dir" 120 6 || { echo "pytest failing or over budget"; return 1; }
	return 0
}

setup_regex() {
	local dir="$1"
	cat >"$dir/test_rx.py" <<'EOF'
# Spec-by-tests: implement rx.py with fullmatch(pattern, text).
# Returns None when the WHOLE text does not match the pattern; on success
# returns a tuple of captured group strings in opening-paren order
# (empty tuple if the pattern has no groups; None for a group that did
# not participate in the match). Quantifiers are greedy with backtracking;
# a repeated group captures its LAST iteration. Using the re/regex modules
# is forbidden — write the engine yourself.
from rx import fullmatch

def test_literal():
    assert fullmatch("abc", "abc") == ()
    assert fullmatch("abc", "abx") is None
    assert fullmatch("abc", "abcd") is None      # fullmatch, not prefix
    assert fullmatch("", "") == ()

def test_dot():
    assert fullmatch("a.c", "abc") == ()
    assert fullmatch("a.c", "a.c") == ()
    assert fullmatch("a.c", "ac") is None

def test_escaped_specials():
    assert fullmatch(r"a\.c", "a.c") == ()
    assert fullmatch(r"a\.c", "abc") is None
    assert fullmatch(r"1\+1", "1+1") == ()
    assert fullmatch(r"\(x\)", "(x)") == ()

def test_char_class():
    assert fullmatch("[abc]", "b") == ()
    assert fullmatch("[abc]", "d") is None
    assert fullmatch("[a-z0-9]", "q") == ()
    assert fullmatch("[a-z0-9]", "7") == ()
    assert fullmatch("[a-z0-9]", "Q") is None

def test_negated_class():
    assert fullmatch("[^0-9]", "a") == ()
    assert fullmatch("[^0-9]", "5") is None

def test_class_escapes():
    assert fullmatch(r"\d\d", "42") == ()
    assert fullmatch(r"\d", "x") is None
    assert fullmatch(r"\w+", "ab_1") == ()
    assert fullmatch(r"a\s+b", "a \t b") == ()

def test_star_greedy_with_backtracking():
    assert fullmatch("a*", "") == ()
    assert fullmatch("a*", "aaaa") == ()
    assert fullmatch("a*a", "aaaa") == ()        # star must give one back
    assert fullmatch("a*b", "b") == ()
    assert fullmatch(".*bc", "abcbc") == ()      # .* must backtrack to last bc

def test_plus_and_question():
    assert fullmatch("ab+c", "abbbc") == ()
    assert fullmatch("ab+c", "ac") is None
    assert fullmatch("colou?r", "color") == ()
    assert fullmatch("colou?r", "colour") == ()

def test_alternation():
    assert fullmatch("cat|dog", "cat") == ()
    assert fullmatch("cat|dog", "dog") == ()
    assert fullmatch("cat|dog", "cow") is None
    assert fullmatch("a(b|c)d", "abd") == ("b",)
    assert fullmatch("a(b|c)d", "acd") == ("c",)
    assert fullmatch("a(b|c)d", "ad") is None

def test_capture_groups():
    assert fullmatch(r"(\d+)-(\d+)", "12-34") == ("12", "34")

def test_repeated_group_captures_last_iteration():
    assert fullmatch("(ab)+", "ababab") == ("ab",)
    assert fullmatch("(ab)+", "aba") is None
    assert fullmatch("(a|b)+c", "abac") == ("a",)

def test_nested_groups():
    assert fullmatch("((a|b)+)c", "abac") == ("aba", "a")

def test_backtracking_across_groups():
    assert fullmatch("(a+)(a+)", "aaa") == ("aa", "a")

def test_optional_group_not_participating():
    assert fullmatch("(ab)?c", "abc") == ("ab",)
    assert fullmatch("(ab)?c", "c") == (None,)
EOF
	sha256sum "$dir/test_rx.py" >"$dir/.protected.sha"
}

prompt_regex() {
	echo "This directory contains test_rx.py, a pytest suite that fully specifies a small regular-expression engine: implement rx.py with a function fullmatch(pattern, text) that matches the ENTIRE text and returns a tuple of captured groups (see the comment at the top of the test file for the exact return contract). Supported syntax: literal characters, '.', backslash escapes of specials, character classes with ranges and negation like [a-z0-9] and [^0-9], the escapes \\d \\w \\s, greedy quantifiers '*' '+' '?' with proper backtracking, alternation '|', and nestable capturing groups '(...)'. You must implement the matching engine yourself: importing or using Python's re/regex modules is forbidden and will fail verification. Do not modify test_rx.py. Run 'python -m pytest -q' and fix your implementation until all tests pass."
}

verify_regex() {
	local dir="$1"
	(cd "$dir" && sha256sum -c .protected.sha >/dev/null 2>&1) || { echo "protected test file modified"; return 1; }
	[[ -f "$dir/rx.py" ]] || { echo "rx.py not created"; return 1; }
	# AST check, not grep: forbid importing re/regex (any form) and dynamic
	# import escape hatches (__import__, importlib)
	local rc=0
	(cd "$dir" && python - <<-'PY'
		import ast, sys
		try:
		    tree = ast.parse(open("rx.py").read())
		except SyntaxError as e:
		    print(f"rx.py does not parse: {e}", file=sys.stderr)
		    sys.exit(2)
		FORBIDDEN = {"re", "regex", "sre_compile", "sre_parse", "_sre", "importlib"}
		bad = []
		for n in ast.walk(tree):
		    if isinstance(n, ast.Import):
		        bad += [a.name for a in n.names if a.name.split(".")[0] in FORBIDDEN]
		    elif isinstance(n, ast.ImportFrom):
		        if n.module and n.module.split(".")[0] in FORBIDDEN:
		            bad.append(n.module)
		    elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
		            and n.func.id == "__import__":
		        bad.append("__import__")
		sys.exit(1 if bad else 0)
	PY
	) || rc=$?
	(( rc == 2 )) && { echo "rx.py has invalid syntax"; return 1; }
	(( rc != 0 )) && { echo "used re/regex/importlib"; return 1; }
	scored_pytest "$dir" 60 14 || { echo "pytest failing"; return 1; }
	return 0
}

# ------------------------------------------------------------- plumbing ----

wait_health() {
	local base i
	base=$("$AILLAMA_BIN" env | sed -n 's/^export OPENAI_BASE_URL="\(.*\)\/v1".*/\1/p')
	[[ -n "$base" ]] || die "cannot determine endpoint from '$AILLAMA_BIN env'"
	for i in $(seq 1 60); do
		curl -sf "$base/health" >/dev/null 2>&1 && return 0
		sleep 5
	done
	die "llama-server not healthy after 300 s"
}

switch_model() {
	local model="$1"
	echo "--- switching llama-server to profile: $model ---"
	local swlog
	swlog=$(mktemp /tmp/bench-agentic-switch.XXXXXX.log)
	"$AILLAMA_BIN" switch "$model" >"$swlog" 2>&1 || {
		tail -5 "$swlog" >&2
		die "aillama switch $model failed (full log: $swlog)"
	}
	wait_health
}

RESULTS=()

run_task() {
	local model="$1" task="$2" dir rc=0 start dur verdict reason="" score=""
	dir="$WORKROOT/$model/$task"
	rm -rf "$dir" && mkdir -p "$dir"
	"setup_$task" "$dir"
	echo "=== [$model/$task] running (timeout ${TASK_TIMEOUT}s, log: $dir/agent.log) ==="
	start=$(date +%s)
	(cd "$dir" && OPENAI_MODEL="$model" timeout "$TASK_TIMEOUT" \
		"$QWEN_BIN" -m "$model" --approval-mode yolo -p "$("prompt_$task")" \
		>"$dir/agent.log" 2>&1) || rc=$?
	dur=$(( $(date +%s) - start ))
	if [[ $rc -eq 124 ]]; then
		verdict="TIMEOUT"
		# score whatever the agent left behind at the cutoff (a 13/14
		# TIMEOUT and an empty directory are very different failures)
		("verify_$task" "$dir" >/dev/null 2>&1) || true
	elif reason=$("verify_$task" "$dir"); then
		verdict="PASS"
	else
		verdict="FAIL"
	fi
	score=$(cat "$dir/.score" 2>/dev/null) || true
	echo "=== [$model/$task] $verdict (${dur}s) ${score:+[$score] }${reason:+— $reason}"
	RESULTS+=("$model|$task|$verdict|${score:--}|${dur}s|$reason")
}

summary() {
	local r
	echo
	echo "==================== SUMMARY ===================="
	printf '%-14s %-10s %-8s %-13s %-7s %s\n' "MODEL" "TASK" "VERDICT" "SCORE" "TIME" "NOTE"
	for r in "${RESULTS[@]}"; do
		IFS='|' read -r m t v s d n <<<"$r"
		printf '%-14s %-10s %-8s %-13s %-7s %s\n' "$m" "$t" "$v" "$s" "$d" "$n"
	done
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && pomoc
command -v "$QWEN_BIN" >/dev/null || die "binary not found: $QWEN_BIN"
command -v "$AILLAMA_BIN" >/dev/null || die "binary not found: $AILLAMA_BIN"
eval "$("$AILLAMA_BIN" env | grep '^export')"

for model in ${MODELS//,/ }; do
	switch_model "$model"
	for task in ${TASKS//,/ }; do
		run_task "$model" "$task"
	done
done
summary
