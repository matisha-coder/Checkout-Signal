"""
train_model.py
Trains a purchase-intent (conversion) prediction model on the
UCI "Online Shoppers Purchasing Intention" dataset and saves the
trained model + feature list to disk for the Streamlit app to use.

Dataset: download 'online_shoppers_intention.csv' from Kaggle/UCI and
place it in this same folder before running this script.
Kaggle search term: "Online Shoppers Purchasing Intention Dataset"
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import joblib

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("online_shoppers_intention.csv")
print("Data shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# 2. Basic preprocessing
# ---------------------------------------------------------
# Encode categorical columns
cat_cols = df.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Target column is 'Revenue' (True/False -> purchased or not)
df["Revenue"] = df["Revenue"].astype(int)

X = df.drop(columns=["Revenue"])
y = df["Revenue"]

feature_names = X.columns.tolist()

# ---------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 4. Train models
# ---------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------
def evaluate(model, name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"\n--- {name} ---")
    print("Accuracy :", round(accuracy_score(y_test, preds), 4))
    print("Precision:", round(precision_score(y_test, preds), 4))
    print("Recall   :", round(recall_score(y_test, preds), 4))
    print("F1 Score :", round(f1_score(y_test, preds), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_test, probs), 4))
    print(classification_report(y_test, preds))

evaluate(log_reg, "Logistic Regression")
evaluate(rf, "Random Forest")

# ---------------------------------------------------------
# 6. Save the better model (Random Forest usually wins here)
# ---------------------------------------------------------
joblib.dump(rf, "model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(feature_names, "feature_names.pkl")

print("\nSaved model.pkl, encoders.pkl, feature_names.pkl")
print("Now run: streamlit run app.py")
