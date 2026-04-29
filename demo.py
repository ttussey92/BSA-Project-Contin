#!/usr/bin/env python3
"""
BSA/AML Transaction Risk Detection — Demo Script

Usage:
    python demo.py
"""

from pathlib import Path

import pandas as pd

from aml_analysis import run_pipeline


def main() -> None:
    print("=" * 70)
    print("BSA/AML Transaction Risk Detection — DEMO")
    print("=" * 70)

    print("\n[1/5] Running AML analysis pipeline...")
    df, results = run_pipeline()

    print("\n[2/5] Loading flagged transactions...")
    flagged_path = Path("flagged_transactions.csv")
    flagged = pd.read_csv(flagged_path)
    print(f"Found {len(flagged)} high/critical risk transactions")

    print("\n[3/5] Top 10 Highest Risk Transactions:")
    print("-" * 80)
    top_10 = flagged.head(10)[[
        "transaction_id",
        "account_id",
        "amount",
        "transaction_type",
        "composite_risk_score",
        "risk_tier",
    ]]
    print(top_10.to_string(index=False, float_format="%.1f"))

    print("\n[4/5] Evaluation Summary:")
    print("-" * 35)
    for method, metrics in results.items():
        print(
            f"{method:11s}: precision={metrics['precision']:.1%}, "
            f"recall={metrics['recall']:.1%}, f1={metrics['f1']:.1%}"
        )

    print("\n[5/5] Rule Trigger Summary:")
    print("-" * 25)
    rule_cols = [
        "rule_structuring",
        "rule_rapid_movement",
        "rule_dormant_spike",
        "rule_round_dollar",
    ]
    rule_names = ["Structuring", "Rapid Movement", "Dormant Spike", "Round Dollar"]
    for rule, name in zip(rule_cols, rule_names):
        count = flagged[rule].sum()
        if count > 0:
            print(f"{name:15s}: {int(count)} triggers")

    print("\n✅ Demo complete!")
    print("📋 flagged_transactions.csv contains the prioritized review queue")
    print("📈 sensitivity_analysis.csv contains model-threshold comparison metrics")
    print("🖼️  aml_visuals.png contains the visual summary")


if __name__ == "__main__":
    main()
