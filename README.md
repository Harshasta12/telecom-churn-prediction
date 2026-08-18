# Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to churn, built to prioritize **catching at-risk customers over avoiding false alarms** — a deliberate choice based on the asymmetric cost of the two mistakes in a retention context.

🔗 **[Live demo](#)** *(add your Streamlit Cloud link here after deploying)*

---

## The problem

Given a telecom customer's account and usage data, predict whether they're likely to churn — so a retention team can reach out before they leave.

**Dataset:** [Churn in Telecoms](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset) — 3,333 customers, 21 features (account info, service plans, usage patterns), ~14.5% churn rate.

## Key findings from EDA

- **International plan is the single strongest churn driver** — customers with one churn at **42.4%**, vs **11.5%** for those without (confirmed with large enough group sizes to trust the gap).
- **Customer service calls matter a lot, and non-linearly** — churn risk jumps sharply at 4+ calls (51.7% churn) vs fewer (11.3%), not a smooth linear increase.
- **Voice mail plan is protective** — users churn at roughly half the rate of non-users (8.7% vs 16.7%).
- **`state` was ruled out as a feature** — despite some states showing much higher raw churn rates, a Monte Carlo simulation showed that with per-state sample sizes as low as 34 customers, seeing *some* state hit an extreme rate by pure chance happens ~30% of the time. The pattern wasn't statistically trustworthy enough to model on.
- **Caught a multicollinearity issue** — `number vmail messages` and `voice mail plan` were 0.96 correlated, which was distorting logistic regression's coefficients (a flipped sign) without actually hurting predictions. Removed the redundant feature.

## Modeling approach

Started with a simple, interpretable baseline and moved to more powerful models only where the added complexity was justified by measurable improvement.

| Model | Recall | Precision | F1 |
|---|---|---|---|
| Logistic Regression (baseline) | 0.20 | 0.41 | 0.27 |
| Logistic Regression (class-weighted) | 0.86 | 0.49 | 0.63 |
| Random Forest | 0.76 | 0.74 | 0.75 |
| **XGBoost (threshold-tuned)** | **0.79** | **0.71** | **0.75** |

**Final model: XGBoost, decision threshold = 0.21** (instead of the default 0.5).

### Why threshold 0.21, and why recall over precision?

In this context, missing a customer who's about to churn is a permanent, silent loss — no chance to intervene. A false alarm just costs a retention team a bit of wasted outreach. Given that asymmetry, the model is tuned to catch ~79% of actual churners, accepting a lower precision (71%) in exchange. The full precision-recall trade-off curve was used to pick a threshold that maximized recall without letting precision collapse — not just recall in isolation, which would make the model unusable (flagging everyone).

## Repo structure

```
├── churn_analysis.ipynb      # Full pipeline: cleaning, EDA, feature engineering, modeling
├── deploy.py                 # Minimal script demonstrating model loading + single prediction
├── app.py                    # Streamlit interactive demo
├── churn_model_xgb.pkl       # Trained XGBoost model
├── churn_scaler.pkl          # Fitted StandardScaler (must be used with the model)
├── model_config.json         # Feature list + decision threshold
└── README.md
```

## Running it locally

```bash
pip install pandas scikit-learn xgboost joblib streamlit

# Interactive demo
streamlit run app.py

# Or run a single prediction from the command line
python deploy.py
```

## What I'd do next

- SHAP values for per-prediction explanations, not just global feature importance
- K-fold cross-validation to check result stability across different splits
- Hyperparameter tuning (currently using XGBoost defaults + class weighting)
- Batch scoring script for evaluating many customers from a CSV at once
