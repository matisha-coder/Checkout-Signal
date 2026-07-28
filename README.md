# Purchase Intent Predictor

Predicts whether a website visitor session will convert into a purchase,
based on session behavior (pages visited, time on site, bounce/exit rate,
visitor type, etc.). Built to mirror real eCommerce conversion-optimization
problems (relevant to checkout/RTO/retention use cases).

## Dataset
UCI / Kaggle **"Online Shoppers Purchasing Intention Dataset"**
(~12,000 sessions, binary target: `Revenue` = purchased or not).

Download `online_shoppers_intention.csv` and place it in this folder.

## Steps to run (local)

```bash
pip install -r requirements.txt

# 1. Train the model (creates model.pkl, encoders.pkl, feature_names.pkl)
python train_model.py

# 2. Launch the demo app
streamlit run app.py
```

## Model & Metrics
Two models are trained and compared: Logistic Regression (baseline) and
Random Forest (final). Evaluated on a held-out 20% test split using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

> Random Forest achieved 88% accuracy and 0.92 ROC-AUC on held-out test data.

DEPLOYED APP LINK :
https://checkout-signal-lufu6pvatw7uu3zjsz9u5u.streamlit.app/
