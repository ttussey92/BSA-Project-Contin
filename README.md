# BSA/AML Transaction Risk Detection

**Applied machine learning and rule-based typology detection for financial crimes compliance.**

A lightweight, interpretable detection pipeline that combines an Isolation Forest anomaly model with FinCEN-aligned typology rules to produce a prioritized SAR review queue. Built as a portfolio project to demonstrate the intersection of BSA/AML domain expertise and quantitative skills for community bank compliance, financial crimes consulting, and model risk management roles.

---

## System Architecture

```
Raw Transactions
       │
       ▼
Feature Engineering (9 BSA-relevant features)
       │
       ├──► Isolation Forest (anomaly score)
       │
       ├──► Rule-Based Alerts (4 typology rules)
       │
       ▼
Composite Risk Score (0–100)
  50% ML + 35% Rules + 15% Country Risk
       │
       ▼
Risk Tiers: Low | Medium | High | Critical
       │
       ▼
Prioritized SAR Review Queue
```

---

## Typologies Covered

| Typology | Regulatory Basis | Detection Method |
|---|---|---|
| Structuring (Smurfing) | 31 CFR § 1020.320 | Rule: 3+ near-$10k cash deposits per account |
| Rapid Movement / Layering | FFIEC BSA/AML Manual | Rule: same-day wire in/out + elevated jurisdiction |
| Dormant Account Spike | FFIEC Unusual Activity | Rule + ML: 90+ day gap + high value + off-hours |
| Round-Dollar Anomaly | FinCEN SAR guidance | Rule + ML: statistical rarity of even amounts |

---

## Results (on synthetic dataset, n=1,000)

| Method | Precision | Recall | F1 |
|---|---|---|---|
| ML Only (Isolation Forest) | 77.0% | 77.0% | 77.0% |
| Rules Only | 100% | 81.0% | 89.5% |
| **Combined (ML + Rules)** | **81.3%** | **100%** | **89.7%** |

The combined system achieves perfect recall (zero missed suspicious transactions) with manageable false-positive volume — a defensible production trade-off under SR 11-7 guidance, where missed SARs carry far higher regulatory cost than investigative overhead.

---

## Regulatory Alignment

- **BSA § 5318(g)** — SAR filing obligations inform typology rule design
- **31 CFR § 1020.320** — CTR threshold and structuring detection
- **SR 11-7** — Model treated as a documented, validated asset; conceptual soundness and ongoing monitoring considerations built into design
- **FFIEC BSA/AML Examination Manual** — Country risk tiers follow FATF blacklist/greylist geography
- **FinCEN Geographic Targeting Orders** — High-risk jurisdiction flags

---

## Tech Stack

- Python 3.11
- scikit-learn (Isolation Forest, StandardScaler)
- pandas, NumPy
- matplotlib, seaborn

---

## Files

| File | Description |
|---|---|
| `aml_analysis.py` | Full pipeline: data gen → features → model → rules → scoring → output |
| `index.html` | Portfolio summary page (served via GitHub Pages) |
| `flagged_transactions.csv` | Sample output: High/Critical tier flagged transactions |
| `aml_visuals.png` | Four-panel visualization of detection results |

---

## Running the Analysis

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python aml_analysis.py
```

Outputs `flagged_transactions.csv` and `aml_visuals.png` to the working directory.

---

## Background

Built as a portfolio project alongside MScFE coursework (Financial Engineering, Deep Learning for Finance). Targets BSA/AML analyst, financial crimes compliance, and model risk management roles at community banks and regional financial institutions.

---

*Synthetic data only. No real customer or transaction data used.*
