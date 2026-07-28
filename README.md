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

Fill in your actual numbers after running `train_model.py`, e.g.:
> Random Forest achieved 90% accuracy and 0.87 ROC-AUC on held-out test data.

## Deploy for free (Streamlit Community Cloud)
1. Push this folder (including `model.pkl`, `encoders.pkl`,
   `feature_names.pkl` generated after training) to a public GitHub repo.
2. Go to https://share.streamlit.io
3. Click **New app** → select your repo → set main file to `app.py` → Deploy.
4. You'll get a public URL (e.g. `https://yourname-purchase-intent.streamlit.app`)
   to put on your resume.

## Resume bullet
> Built and deployed a purchase-intent prediction model (Random Forest)
> on eCommerce session-behavior data, achieving [X]% accuracy and [X] ROC-AUC;
> deployed via Streamlit for real-time conversion-likelihood scoring.
