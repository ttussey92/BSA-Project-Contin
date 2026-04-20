"""
BSA/AML Transaction Risk Detection System
==========================================

Combines Isolation Forest anomaly detection with FinCEN-aligned rule-based
typology alerts to produce a prioritized SAR review queue with a
composite 0-100 risk score.

Regulatory alignment:
  - BSA § 5318(g):        SAR filing obligations
  - 31 CFR § 1020.320:    CTR threshold / structuring detection
  - SR 11-7:              Model risk management (documentation, benchmarking)
  - FFIEC BSA/AML Examination Manual
  - FinCEN Geographic Targeting Orders

Data: Synthetic only. No real customer or transaction data used.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ---------------------------------------------------------------------------

def generate_synthetic_transactions(n_normal: int = 900, n_suspicious: int = 100) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset embedding four FinCEN typologies:
      - Structuring (smurfing)
      - Rapid movement / layering
      - Dormant account spikes
      - Round-dollar anomalies
    """
    # --- Normal transactions ---
    normal = pd.DataFrame({
        "transaction_id": [f"TXN{i:06d}" for i in range(n_normal)],
        "account_id": np.random.randint(1000, 1100, n_normal),
        "amount": np.abs(np.random.lognormal(mean=6.0, sigma=1.2, size=n_normal)),
        "transaction_type": np.random.choice(
            ["deposit", "withdrawal", "wire_in", "wire_out", "ach"],
            size=n_normal,
            p=[0.35, 0.30, 0.10, 0.10, 0.15],
        ),
        "hour": np.random.choice(range(6, 20), n_normal),
        "day_of_week": np.random.choice(range(7), n_normal),
        "days_since_last_txn": np.random.exponential(scale=2.0, size=n_normal).astype(int),
        "counterparty_country_risk": np.random.choice(
            [1, 2, 3], size=n_normal, p=[0.85, 0.12, 0.03]
        ),
        "is_suspicious": 0,
    })

    suspicious_rows = []
    per_type = n_suspicious // 4

    # Structuring: cash deposits just below $10k CTR threshold
    for _ in range(per_type):
        suspicious_rows.append({
            "transaction_id": f"TXN_SUS_S{_:04d}",
            "account_id": np.random.randint(2000, 2020),
            "amount": np.random.uniform(9000, 9900),
            "transaction_type": "deposit",
            "hour": np.random.choice(range(8, 18)),
            "day_of_week": np.random.choice(range(5)),
            "days_since_last_txn": np.random.randint(0, 2),
            "counterparty_country_risk": 1,
            "is_suspicious": 1,
        })

    # Rapid movement: same-day wire in/out to elevated-risk jurisdictions
    for _ in range(per_type):
        suspicious_rows.append({
            "transaction_id": f"TXN_SUS_R{_:04d}",
            "account_id": np.random.randint(2020, 2040),
            "amount": np.random.uniform(25000, 90000),
            "transaction_type": np.random.choice(["wire_in", "wire_out"]),
            "hour": np.random.choice(range(9, 17)),
            "day_of_week": np.random.choice(range(5)),
            "days_since_last_txn": 0,
            "counterparty_country_risk": np.random.choice([2, 3], p=[0.4, 0.6]),
            "is_suspicious": 1,
        })

    # Dormant account spike: long gap + high value + off-hours
    for _ in range(per_type):
        suspicious_rows.append({
            "transaction_id": f"TXN_SUS_D{_:04d}",
            "account_id": np.random.randint(2040, 2060),
            "amount": np.random.uniform(15000, 75000),
            "transaction_type": np.random.choice(["withdrawal", "wire_out"]),
            "hour": np.random.choice([1, 2, 3, 4, 22, 23]),
            "day_of_week": np.random.choice([5, 6]),
            "days_since_last_txn": np.random.randint(90, 365),
            "counterparty_country_risk": np.random.choice([1, 2]),
            "is_suspicious": 1,
        })

    # Round-dollar anomaly
    for _ in range(per_type):
        suspicious_rows.append({
            "transaction_id": f"TXN_SUS_Z{_:04d}",
            "account_id": np.random.randint(2060, 2080),
            "amount": np.random.choice([50000, 75000, 100000, 25000]),
            "transaction_type": "wire_out",
            "hour": np.random.choice(range(9, 17)),
            "day_of_week": np.random.choice(range(5)),
            "days_since_last_txn": np.random.randint(0, 10),
            "counterparty_country_risk": np.random.choice([1, 2, 3]),
            "is_suspicious": 1,
        })

    df = pd.concat([normal, pd.DataFrame(suspicious_rows)], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount_log"] = np.log1p(df["amount"])
    df["is_near_ctr"] = ((df["amount"] >= 9000) & (df["amount"] < 10000)).astype(int)
    df["is_round_dollar"] = (df["amount"] % 1000 == 0).astype(int)
    df["is_off_hours"] = ((df["hour"] < 6) | (df["hour"] > 20)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_dormant_reactivation"] = (df["days_since_last_txn"] > 90).astype(int)

    type_dummies = pd.get_dummies(df["transaction_type"], prefix="txn")
    df = pd.concat([df, type_dummies], axis=1)
    return df


# ---------------------------------------------------------------------------
# 3. ISOLATION FOREST — UNSUPERVISED ANOMALY DETECTION
# ---------------------------------------------------------------------------

def run_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "amount_log", "is_near_ctr", "is_round_dollar", "is_off_hours",
        "is_weekend", "is_dormant_reactivation", "days_since_last_txn",
        "counterparty_country_risk", "hour",
    ]
    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        n_estimators=200, contamination=0.10,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_scaled)

    raw = model.decision_function(X_scaled)
    df["ml_anomaly_score"] = 100 * (raw.max() - raw) / (raw.max() - raw.min())
    df["ml_flag"] = (model.predict(X_scaled) == -1).astype(int)
    return df


# ---------------------------------------------------------------------------
# 4. RULE-BASED TYPOLOGY ALERTS
# ---------------------------------------------------------------------------

def apply_typology_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rule 1 — Structuring
    struct_counts = (
        df[df["is_near_ctr"] == 1]
        .groupby("account_id").size()
        .reset_index(name="near_ctr_count")
    )
    struct_accounts = struct_counts[struct_counts["near_ctr_count"] >= 3]["account_id"].tolist()
    df["rule_structuring"] = (
        (df["account_id"].isin(struct_accounts)) & (df["is_near_ctr"] == 1)
    ).astype(int)

    # Rule 2 — Rapid movement
    df["rule_rapid_movement"] = (
        (df["transaction_type"].isin(["wire_in", "wire_out"]))
        & (df["amount"] > 10000)
        & (df["days_since_last_txn"] <= 1)
        & (df["counterparty_country_risk"] >= 2)
    ).astype(int)

    # Rule 3 — Dormant spike
    df["rule_dormant_spike"] = (
        (df["is_dormant_reactivation"] == 1) & (df["amount"] > 10000)
    ).astype(int)

    # Rule 4 — Round-dollar anomaly
    df["rule_round_dollar"] = (
        (df["is_round_dollar"] == 1) & (df["amount"] >= 25000)
    ).astype(int)

    rule_cols = ["rule_structuring", "rule_rapid_movement",
                 "rule_dormant_spike", "rule_round_dollar"]
    df["rule_flag_count"] = df[rule_cols].sum(axis=1)
    df["rule_flag"] = (df["rule_flag_count"] > 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# 5. COMPOSITE RISK SCORE
# ---------------------------------------------------------------------------

def compute_composite_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    country_norm = (df["counterparty_country_risk"] - 1) * 50  # 0, 50, 100
    rule_norm = np.clip(df["rule_flag_count"] * 33, 0, 100)

    df["composite_risk_score"] = (
        0.50 * df["ml_anomaly_score"]
        + 0.35 * rule_norm
        + 0.15 * country_norm
    ).round(1)

    def tier(s):
        if s >= 75: return "Critical"
        if s >= 55: return "High"
        if s >= 35: return "Medium"
        return "Low"
    df["risk_tier"] = df["composite_risk_score"].apply(tier)
    return df


# ---------------------------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------------------------

def evaluate(df: pd.DataFrame) -> dict:
    def metrics(y_true, y_pred):
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        return {"precision": precision, "recall": recall, "f1": f1,
                "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    combined_flag = ((df["ml_flag"] == 1) | (df["rule_flag"] == 1)).astype(int)
    return {
        "ml_only":     metrics(df["is_suspicious"], df["ml_flag"]),
        "rules_only":  metrics(df["is_suspicious"], df["rule_flag"]),
        "combined":    metrics(df["is_suspicious"], combined_flag),
    }


# ---------------------------------------------------------------------------
# 7. VISUALS
# ---------------------------------------------------------------------------

def make_visuals(df: pd.DataFrame, outpath: str = "aml_visuals.png") -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BSA/AML Transaction Risk Detection — Results",
                 fontsize=15, fontweight="bold")

    # Panel 1: risk score distribution
    ax = axes[0, 0]
    for label, sub in df.groupby("is_suspicious"):
        ax.hist(sub["composite_risk_score"], bins=30, alpha=0.6,
                label="Suspicious" if label == 1 else "Normal")
    ax.set_title("Composite Risk Score Distribution")
    ax.set_xlabel("Risk Score (0-100)")
    ax.legend()

    # Panel 2: ML vs Rules overlap
    ax = axes[0, 1]
    overlap = pd.crosstab(df["ml_flag"], df["rule_flag"])
    sns.heatmap(overlap, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("ML Flag vs Rule Flag (Detection Overlap)")
    ax.set_xlabel("Rule Flag"); ax.set_ylabel("ML Flag")

    # Panel 3: Amount by transaction type
    ax = axes[1, 0]
    df_plot = df.copy()
    df_plot["status"] = df_plot["is_suspicious"].map({0: "Normal", 1: "Suspicious"})
    sns.boxplot(data=df_plot, x="transaction_type", y="amount",
                hue="status", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Transaction Amount by Type (log)")
    ax.tick_params(axis="x", rotation=30)

    # Panel 4: Risk tier counts
    ax = axes[1, 1]
    tier_order = ["Low", "Medium", "High", "Critical"]
    tier_counts = df["risk_tier"].value_counts().reindex(tier_order).fillna(0)
    colors = ["#10b981", "#f59e0b", "#ef4444", "#991b1b"]
    ax.bar(tier_counts.index, tier_counts.values, color=colors)
    ax.set_title("Transactions by Risk Tier")
    ax.set_ylabel("Count")
    for i, v in enumerate(tier_counts.values):
        ax.text(i, v + 5, str(int(v)), ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("BSA/AML Transaction Risk Detection System")
    print("=" * 70)

    print("\n[1/6] Generating synthetic transactions...")
    df = generate_synthetic_transactions(n_normal=900, n_suspicious=100)
    print(f"  Total: {len(df)}  |  Suspicious (ground truth): {df['is_suspicious'].sum()}")

    print("\n[2/6] Engineering features...")
    df = engineer_features(df)

    print("\n[3/6] Running Isolation Forest...")
    df = run_isolation_forest(df)

    print("\n[4/6] Applying typology rules...")
    df = apply_typology_rules(df)

    print("\n[5/6] Computing composite risk score...")
    df = compute_composite_risk(df)

    print("\n[6/6] Evaluating & exporting...")
    results = evaluate(df)
    print("\n  Method      | Precision |  Recall   |    F1")
    print("  " + "-" * 48)
    for name, m in results.items():
        print(f"  {name:11s} |  {m['precision']:6.1%}   |  {m['recall']:6.1%}  |  {m['f1']:6.1%}")

    # Export flagged
    flagged = df[df["risk_tier"].isin(["High", "Critical"])].sort_values(
        "composite_risk_score", ascending=False
    )
    export_cols = [
        "transaction_id", "account_id", "amount", "transaction_type",
        "composite_risk_score", "risk_tier", "ml_anomaly_score",
        "rule_structuring", "rule_rapid_movement", "rule_dormant_spike",
        "rule_round_dollar", "counterparty_country_risk",
    ]
    flagged[export_cols].to_csv("flagged_transactions.csv", index=False)
    print(f"\n  Exported {len(flagged)} High/Critical transactions to flagged_transactions.csv")

    make_visuals(df)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
