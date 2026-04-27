#!/usr/bin/env python3
"""
BSA/AML Transaction Risk Detection — Demo Script
===============================================

This script demonstrates the AML detection system by:
1. Running the analysis pipeline
2. Displaying key metrics and results
3. Showing the top flagged transactions
4. Opening the visualization

Usage: python demo.py
"""

import pandas as pd
import subprocess
import sys
import os

def main():
    print("=" * 70)
    print("BSA/AML Transaction Risk Detection — DEMO")
    print("=" * 70)

    # Check if results exist, if not run the analysis
    if not os.path.exists("flagged_transactions.csv"):
        print("\n[1/4] Running AML analysis pipeline...")
        try:
            subprocess.run([sys.executable, "aml_analysis.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running analysis: {e}")
            return
    else:
        print("\n[1/4] Using existing analysis results...")

    # Load and display results
    print("\n[2/4] Loading flagged transactions...")
    try:
        flagged = pd.read_csv("flagged_transactions.csv")
        print(f"Found {len(flagged)} high/critical risk transactions")
    except FileNotFoundError:
        print("Error: flagged_transactions.csv not found")
        return

    print("\n[3/4] Top 10 Highest Risk Transactions:")
    print("-" * 80)
    top_10 = flagged.head(10)[[
        'transaction_id', 'account_id', 'amount', 'transaction_type',
        'composite_risk_score', 'risk_tier'
    ]]
    print(top_10.to_string(index=False, float_format='%.1f'))

    print("\n[4/4] Risk Tier Distribution:")
    print("-" * 30)
    tier_counts = flagged['risk_tier'].value_counts()
    for tier, count in tier_counts.items():
        print(f"{tier:8s}: {count} transactions")

    print("\n[5/5] Rule Trigger Summary:")
    print("-" * 25)
    rule_cols = ['rule_structuring', 'rule_rapid_movement', 'rule_dormant_spike', 'rule_round_dollar']
    rule_names = ['Structuring', 'Rapid Movement', 'Dormant Spike', 'Round Dollar']
    for rule, name in zip(rule_cols, rule_names):
        count = flagged[rule].sum()
        if count > 0:
            print(f"{name:15s}: {int(count)} triggers")

    print("\n✅ Demo complete!")
    print("📊 Check aml_visuals.png for detailed charts")
    print("📋 See flagged_transactions.csv for full results")

    # Try to open the visualization if possible
    if os.path.exists("aml_visuals.png"):
        print("🖼️  Visualization saved as aml_visuals.png")
        # On Linux, try to open with default image viewer
        try:
            subprocess.run(["xdg-open", "aml_visuals.png"], check=False)
        except:
            pass  # Ignore if no viewer available

if __name__ == "__main__":
    main()