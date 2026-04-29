"""
BSA/AML Transaction Risk Detection System
==========================================

Combines Isolation Forest anomaly detection with FinCEN-aligned rule-based
typology alerts to produce a prioritized SAR review queue with a composite
0-100 risk score.

Data: Synthetic only. No real customer or transaction data used.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURE_COLS = [
    "amount_log",
    "is_near_ctr",
    "is_round_dollar",
    "is_off_hours",
    "is_weekend",
    "is_dormant_reactivation",
    "days_since_last_txn",
    "counterparty_country_risk",
    "hour",
]

RULE_COLS = [
    "rule_structuring",
    "rule_rapid_movement",
    "rule_dormant_spike",
    "rule_round_dollar",
]


# ---------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ---------------------------------------------------------------------------

def _split_suspicious_counts(n_suspicious: int) -> list[int]:
    """Split requested suspicious rows across the four embedded typologies."""
    base, remainder = divmod(n_suspicious, 4)
    return [base + (1 if i < remainder else 0) for i in range(4)]


def generate_synthetic_transactions(
    n_normal: int = 900,
    n_suspicious: int = 100,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset embedding four BSA/AML typologies:
      - Structuring (smurfing)
      - Rapid movement / layering
      - Dormant account spikes
      - Round-dollar anomalies
    """
    if n_normal < 0 or n_suspicious < 0:
        raise ValueError("n_normal and n_suspicious must be non-negative")

    rng = np.random.default_rng(random_seed)

    normal = pd.DataFrame({
        "transaction_id": [f"TXN{i:06d}" for i in range(n_normal)],
        "account_id": rng.integers(1000, 1100, n_normal),
        "amount": np.abs(rng.lognormal(mean=6.0, sigma=1.2, size=n_normal)),
        "transaction_type": rng.choice(
            ["deposit", "withdrawal", "wire_in", "wire_out", "ach"],
            size=n_normal,
            p=[0.35, 0.30, 0.10, 0.10, 0.15],
        ),
        "hour": rng.choice(range(6, 20), n_normal),
        "day_of_week": rng.choice(range(7), n_normal),
        "days_since_last_txn": rng.exponential(scale=2.0, size=n_normal).astype(int),
        "counterparty_country_risk": rng.choice([1, 2, 3], size=n_normal, p=[0.85, 0.12, 0.03]),
        "is_suspicious": 0,
    })

    suspicious_rows = []
    structuring_n, rapid_n, dormant_n, round_n = _split_suspicious_counts(n_suspicious)

    for i in range(structuring_n):
        suspicious_rows.append({
            "transaction_id": f"TXN_SUS_S{i:04d}",
            "account_id": rng.integers(2000, 2020),
            "amount": rng.uniform(9000, 9900),
            "transaction_type": "deposit",
            "hour": rng.choice(range(8, 18)),
            "day_of_week": rng.choice(range(5)),
            "days_since_last_txn": rng.integers(0, 2),
            "counterparty_country_risk": 1,
            "is_suspicious": 1,
        })

    for i in range(rapid_n):
        suspicious_rows.append({
            "transaction_id": f"TXN_SUS_R{i:04d}",
            "account_id": rng.integers(2020, 2040),
            "amount": rng.uniform(25000, 90000),
            "transaction_type": rng.choice(["wire_in", "wire_out"]),
            "hour": rng.choice(range(9, 17)),
            "day_of_week": rng.choice(range(5)),
            "days_since_last_txn": 0,
            "counterparty_country_risk": rng.choice([2, 3], p=[0.4, 0.6]),
            "is_suspicious": 1,
        })

    for i in range(dormant_n):
        suspicious_rows.append({
            "transaction_id": f"TXN_SUS_D{i:04d}",
            "account_id": rng.integers(2040, 2060),
            "amount": rng.uniform(15000, 75000),
            "transaction_type": rng.choice(["withdrawal", "wire_out"]),
            "hour": rng.choice([1, 2, 3, 4, 22, 23]),
            "day_of_week": rng.choice([5, 6]),
            "days_since_last_txn": rng.integers(90, 365),
            "counterparty_country_risk": rng.choice([1, 2]),
            "is_suspicious": 1,
        })

    for i in range(round_n):
        suspicious_rows.append({
            "transaction_id": f"TXN_SUS_Z{i:04d}",
            "account_id": rng.integers(2060, 2080),
            "amount": rng.choice([25000, 50000, 75000, 100000]),
            "transaction_type": "wire_out",
            "hour": rng.choice(range(9, 17)),
            "day_of_week": rng.choice(range(5)),
            "days_since_last_txn": rng.integers(0, 10),
            "counterparty_country_risk": rng.choice([1, 2, 3]),
            "is_suspicious": 1,
        })

    df = pd.concat([normal, pd.DataFrame(suspicious_rows)], ignore_index=True)
    return df.sample(frac=1, random_state=random_seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount_log"] = np.log1p(df["amount"])
    df["is_near_ctr"] = ((df["amount"] >= 9000) & (df["amount"] < 10000)).astype(int)
    df["is_round_dollar"] = np.isclose(df["amount"] % 1000, 0).astype(int)
    df["is_off_hours"] = ((df["hour"] < 6) | (df["hour"] > 20)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_dormant_reactivation"] = (df["days_since_last_txn"] >= 90).astype(int)

    type_dummies = pd.get_dummies(df["transaction_type"], prefix="txn")
    return pd.concat([df, type_dummies], axis=1)


# ---------------------------------------------------------------------------
# 3. ISOLATION FOREST — UNSUPERVISED ANOMALY DETECTION
# ---------------------------------------------------------------------------

def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.10) -> pd.DataFrame:
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")

    df = df.copy()
    X = df[FEATURE_COLS].values
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    raw = model.decision_function(X_scaled)
    denom = raw.max() - raw.min()
    df["ml_anomaly_score"] = 0 if denom == 0 else 100 * (raw.max() - raw) / denom
    df["ml_flag"] = (model.predict(X_scaled) == -1).astype(int)
    return df


# ---------------------------------------------------------------------------
# 4. RULE-BASED TYPOLOGY ALERTS
# ---------------------------------------------------------------------------

def apply_typology_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    struct_counts = (
        df[df["is_near_ctr"] == 1]
        .groupby("account_id")
        .size()
        .reset_index(name="near_ctr_count")
    )
    struct_accounts = struct_counts[struct_counts["near_ctr_count"] >= 3]["account_id"].tolist()
    df["rule_structuring"] = (
        (df["account_id"].isin(struct_accounts)) & (df["is_near_ctr"] == 1)
    ).astype(int)

    df["rule_rapid_movement"] = (
        (df["transaction_type"].isin(["wire_in", "wire_out"]))
        & (df["amount"] > 10000)
        & (df["days_since_last_txn"] <= 1)
        & (df["counterparty_country_risk"] >= 2)
    ).astype(int)

    df["rule_dormant_spike"] = (
        (df["is_dormant_reactivation"] == 1) & (df["amount"] > 10000)
    ).astype(int)

    df["rule_round_dollar"] = (
        (df["is_round_dollar"] == 1) & (df["amount"] >= 25000)
    ).astype(int)

    df["rule_flag_count"] = df[RULE_COLS].sum(axis=1)
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

    bins = [-np.inf, 35, 55, 75, np.inf]
    labels = ["Low", "Medium", "High", "Critical"]
    df["risk_tier"] = pd.cut(df["composite_risk_score"], bins=bins, labels=labels, right=False).astype(str)
    return df


# ---------------------------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------------------------

def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate(df: pd.DataFrame) -> dict:
    combined_flag = ((df["ml_flag"] == 1) | (df["rule_flag"] == 1)).astype(int)
    return {
        "ml_only": _metrics(df["is_suspicious"], df["ml_flag"]),
        "rules_only": _metrics(df["is_suspicious"], df["rule_flag"]),
        "combined": _metrics(df["is_suspicious"], combined_flag),
    }


def results_to_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for method, metrics in results.items():
        row = {"method": method}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def run_sensitivity_analysis(
    engineered_df: pd.DataFrame,
    contamination_grid: Iterable[float] = (0.03, 0.05, 0.10, 0.15),
) -> pd.DataFrame:
    """Evaluate the ML+rules pipeline across several Isolation Forest settings."""
    rows = []
    for contamination in contamination_grid:
        scored = run_isolation_forest(engineered_df, contamination=contamination)
        scored = apply_typology_rules(scored)
        scored = compute_composite_risk(scored)
        combined = evaluate(scored)["combined"]
        rows.append({
            "contamination": contamination,
            "precision": combined["precision"],
            "recall": combined["recall"],
            "f1": combined["f1"],
            "true_positives": combined["tp"],
            "false_positives": combined["fp"],
            "false_negatives": combined["fn"],
            "high_critical_count": int(scored["risk_tier"].isin(["High", "Critical"]).sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. VISUALS
# ---------------------------------------------------------------------------

def make_visuals(df: pd.DataFrame, outpath: str = "aml_visuals.png") -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BSA/AML Transaction Risk Detection — Results", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    for label, sub in df.groupby("is_suspicious"):
        ax.hist(sub["composite_risk_score"], bins=30, alpha=0.6, label="Suspicious" if label == 1 else "Normal")
    ax.set_title("Composite Risk Score Distribution")
    ax.set_xlabel("Risk Score (0-100)")
    ax.legend()

    ax = axes[0, 1]
    overlap = pd.crosstab(df["ml_flag"], df["rule_flag"])
    sns.heatmap(overlap, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("ML Flag vs Rule Flag (Detection Overlap)")
    ax.set_xlabel("Rule Flag")
    ax.set_ylabel("ML Flag")

    ax = axes[1, 0]
    df_plot = df.copy()
    df_plot["status"] = df_plot["is_suspicious"].map({0: "Normal", 1: "Suspicious"})
    sns.boxplot(data=df_plot, x="transaction_type", y="amount", hue="status", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Transaction Amount by Type (log)")
    ax.tick_params(axis="x", rotation=30)

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
# PIPELINE + MAIN
# ---------------------------------------------------------------------------

def run_pipeline(
    n_normal: int = 900,
    n_suspicious: int = 100,
    output_dir: str | Path = ".",
    export: bool = True,
    make_plots: bool = True,
) -> tuple[pd.DataFrame, dict]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_transactions(n_normal=n_normal, n_suspicious=n_suspicious)
    df = engineer_features(df)
    engineered = df.copy()
    df = run_isolation_forest(df)
    df = apply_typology_rules(df)
    df = compute_composite_risk(df)
    results = evaluate(df)

    if export:
        flagged = df[df["risk_tier"].isin(["High", "Critical"])].sort_values(
            "composite_risk_score", ascending=False
        )
        export_cols = [
            "transaction_id",
            "account_id",
            "amount",
            "transaction_type",
            "composite_risk_score",
            "risk_tier",
            "ml_anomaly_score",
            "rule_structuring",
            "rule_rapid_movement",
            "rule_dormant_spike",
            "rule_round_dollar",
            "counterparty_country_risk",
        ]
        flagged[export_cols].to_csv(output_path / "flagged_transactions.csv", index=False)
        run_sensitivity_analysis(engineered).to_csv(output_path / "sensitivity_analysis.csv", index=False)

    if make_plots:
        make_visuals(df, str(output_path / "aml_visuals.png"))

    return df, results


def main() -> None:
    print("=" * 70)
    print("BSA/AML Transaction Risk Detection System")
    print("=" * 70)

    print("\n[1/6] Running end-to-end analysis pipeline...")
    df, results = run_pipeline()

    print("\n[2/6] Evaluation results")
    print("\n  Method      | Precision |  Recall   |    F1")
    print("  " + "-" * 48)
    for name, m in results.items():
        print(f"  {name:11s} |  {m['precision']:6.1%}   |  {m['recall']:6.1%}  |  {m['f1']:6.1%}")

    flagged_count = int(df["risk_tier"].isin(["High", "Critical"]).sum())
    print(f"\n[3/6] Exported {flagged_count} High/Critical transactions to flagged_transactions.csv")
    print("[4/6] Exported sensitivity analysis to sensitivity_analysis.csv")
    print("[5/6] Saved visualization to aml_visuals.png")
    print("[6/6] Done.\n")


if __name__ == "__main__":
    main()
