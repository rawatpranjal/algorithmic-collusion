#!/usr/bin/env python3
"""
Direct unit tests for the run_auction() function from exp4a and exp4b.

Tests payment rules (FPA vs SPA), reserve price filtering, no-sale outcomes,
and verifies both modules produce identical results.

Usage:
    cd /Users/pranjal/Code/algorithmic-collusion
    PYTHONPATH=src python3 scripts/verification/test_payment_rules_exp4.py
"""

import sys
import numpy as np

from experiments.exp4a import run_auction as run_auction_4a
from experiments.exp4b import run_auction as run_auction_4b


# ── Test case definitions ────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "FPA basic (2 bidders)",
        "bids": [0.8, 0.5],
        "vals": [1.0, 0.9],
        "auction_type": "first",
        "reserve_price": 0.0,
        "expected_winner": 0,
        "expected_payment": 0.8,
        "expected_rewards": [0.2, 0.0],
    },
    {
        "name": "SPA basic (2 bidders)",
        "bids": [0.8, 0.5],
        "vals": [1.0, 0.9],
        "auction_type": "second",
        "reserve_price": 0.0,
        "expected_winner": 0,
        "expected_payment": 0.5,
        "expected_rewards": [0.5, 0.0],
    },
    {
        "name": "Reserve filtering (no valid bids)",
        "bids": [0.2, 0.1],
        "vals": [1.0, 0.9],
        "auction_type": "first",
        "reserve_price": 0.3,
        "expected_winner": -1,
        "expected_payment": 0.0,
        "expected_rewards": [0.0, 0.0],
    },
    {
        "name": "Reserve with 1 valid (SPA pays reserve)",
        "bids": [0.5, 0.1],
        "vals": [1.0, 0.9],
        "auction_type": "second",
        "reserve_price": 0.3,
        "expected_winner": 0,
        "expected_payment": 0.3,
        "expected_rewards": [0.7, 0.0],
    },
    {
        "name": "SPA 4-bidder",
        "bids": [0.8, 0.6, 0.4, 0.3],
        "vals": [1.0, 0.9, 0.8, 0.7],
        "auction_type": "second",
        "reserve_price": 0.0,
        "expected_winner": 0,
        "expected_payment": 0.6,
        "expected_rewards": [0.4, 0.0, 0.0, 0.0],
    },
    {
        "name": "FPA 4-bidder",
        "bids": [0.8, 0.6, 0.4, 0.3],
        "vals": [1.0, 0.9, 0.8, 0.7],
        "auction_type": "first",
        "reserve_price": 0.0,
        "expected_winner": 0,
        "expected_payment": 0.8,
        "expected_rewards": [0.2, 0.0, 0.0, 0.0],
    },
    {
        "name": "SPA reserve + 2 valid (3 bidders)",
        "bids": [0.5, 0.4, 0.1],
        "vals": [1.0, 0.9, 0.8],
        "auction_type": "second",
        "reserve_price": 0.3,
        "expected_winner": 0,
        "expected_payment": 0.4,
        "expected_rewards": [0.6, 0.0, 0.0],
    },
    {
        "name": "No-sale with zero bids",
        "bids": [0.0, 0.0],
        "vals": [1.0, 0.9],
        "auction_type": "first",
        "reserve_price": 0.0,
        "expected_winner": -1,
        "expected_payment": 0.0,
        "expected_rewards": [0.0, 0.0],
    },
]


# ── Test runner ──────────────────────────────────────────────────────────

def run_single_test(test, auction_fn, label):
    """Run one test case against one auction function. Returns (pass, message)."""
    bids = np.array(test["bids"], dtype=np.float64)
    vals = np.array(test["vals"], dtype=np.float64)
    winner, payment, rewards = auction_fn(
        bids, vals, test["auction_type"], test["reserve_price"]
    )

    errors = []

    if winner != test["expected_winner"]:
        errors.append(
            f"  winner: expected {test['expected_winner']}, got {winner}"
        )

    if not np.isclose(payment, test["expected_payment"], atol=1e-9):
        errors.append(
            f"  payment: expected {test['expected_payment']}, got {payment}"
        )

    expected_rewards = np.array(test["expected_rewards"], dtype=np.float64)
    if not np.allclose(rewards, expected_rewards, atol=1e-9):
        errors.append(
            f"  rewards: expected {expected_rewards.tolist()}, got {rewards.tolist()}"
        )

    if errors:
        detail = "\n".join(errors)
        return False, f"FAIL [{label}] {test['name']}\n{detail}"
    else:
        return True, f"PASS [{label}] {test['name']}"


def run_identity_check(test):
    """Verify exp4a and exp4b produce identical results for the same inputs."""
    bids = np.array(test["bids"], dtype=np.float64)
    vals = np.array(test["vals"], dtype=np.float64)

    w_a, p_a, r_a = run_auction_4a(
        bids.copy(), vals.copy(), test["auction_type"], test["reserve_price"]
    )
    w_b, p_b, r_b = run_auction_4b(
        bids.copy(), vals.copy(), test["auction_type"], test["reserve_price"]
    )

    errors = []
    if w_a != w_b:
        errors.append(f"  winner: exp4a={w_a}, exp4b={w_b}")
    if not np.isclose(p_a, p_b, atol=1e-12):
        errors.append(f"  payment: exp4a={p_a}, exp4b={p_b}")
    if not np.allclose(r_a, r_b, atol=1e-12):
        errors.append(f"  rewards: exp4a={r_a.tolist()}, exp4b={r_b.tolist()}")

    if errors:
        detail = "\n".join(errors)
        return False, f"FAIL [identity] {test['name']}\n{detail}"
    else:
        return True, f"PASS [identity] {test['name']}"


def main():
    np.random.seed(42)  # Deterministic tie-breaking

    passed = 0
    failed = 0
    total = 0
    messages = []

    print("=" * 70)
    print("Payment Rules Test: run_auction() from exp4a and exp4b")
    print("=" * 70)
    print()

    # Phase 1: Correctness tests against both modules
    for module_label, auction_fn in [("exp4a", run_auction_4a), ("exp4b", run_auction_4b)]:
        print(f"--- Testing {module_label} ---")
        for test in TEST_CASES:
            total += 1
            try:
                ok, msg = run_single_test(test, auction_fn, module_label)
            except Exception as e:
                ok = False
                msg = f"FAIL [{module_label}] {test['name']}\n  Exception: {e}"
            if ok:
                passed += 1
            else:
                failed += 1
            messages.append(msg)
            print(msg)
        print()

    # Phase 2: Identity checks (exp4a == exp4b)
    print("--- Identity checks (exp4a == exp4b) ---")
    for test in TEST_CASES:
        total += 1
        try:
            ok, msg = run_identity_check(test)
        except Exception as e:
            ok = False
            msg = f"FAIL [identity] {test['name']}\n  Exception: {e}"
        if ok:
            passed += 1
        else:
            failed += 1
        messages.append(msg)
        print(msg)

    # Summary
    print()
    print("=" * 70)
    print(f"SUMMARY: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 70)

    if failed > 0:
        print("\nFailed tests:")
        for msg in messages:
            if msg.startswith("FAIL"):
                print(f"  {msg.splitlines()[0]}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
