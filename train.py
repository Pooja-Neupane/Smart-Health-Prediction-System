# train.py
"""
Train a disease prediction model based on symptom inputs.
Expected dataset format (preferred): CSV with columns:
- disease (string)
- symptoms (string of comma-separated symptoms), e.g. "fever,cough,sore throat"

If such a file named 'symptoms_disease.csv' is not found, the script will create
a small synthetic dataset so you can proceed end-to-end.

Outputs saved to disk:
- model.joblib        : trained sklearn RandomForestClassifier
- label_encoder.joblib: LabelEncoder for disease labels
- features.json       : list of all symptoms used as features (order matters)
- precautions.json    : mapping disease -> precaution text (generic if not supplied)
"""

import os
import json
import joblib
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------
# Parameters
# -------------------------
DATA_FILE = "symptoms_disease.csv"   # user dataset (optional)
MODEL_FILE = "model.joblib"
LABEL_ENCODER_FILE = "label_encoder.joblib"
FEATURES_FILE = "features.json"
PRECAUTIONS_FILE = "precautions.json"
RANDOM_STATE = 42

# -------------------------
# Utility: create synthetic dataset (if no dataset provided)
# -------------------------
def create_synthetic_dataset(path="symptoms_disease.csv", n_samples=800):
    diseases = {
        "Common Cold": ["cough", "sore throat", "runny nose", "sneezing", "mild fever"],
        "Flu": ["fever", "chills", "headache", "muscle pain", "fatigue", "cough"],
        "Migraine": ["headache", "nausea", "light sensitivity", "aura"],
        "Food Poisoning": ["nausea", "vomiting", "diarrhea", "stomach pain", "fever"],
        "Allergy": ["sneezing", "runny nose", "itchy eyes", "rash", "congestion"],
        "COVID-19": ["fever", "dry cough", "fatigue", "loss of smell", "difficulty breathing"],
        "Stomach Flu": ["diarrhea", "nausea", "vomiting", "stomach pain", "fever"],
    }
    rows = []
    disease_names = list(diseases.keys())
    for _ in range(n_samples):
        d = random.choice(disease_names)
        base_symptoms = diseases[d]
        # choose subset and add some noise symptoms
        k = max(1, int(random.gauss(len(base_symptoms), 1)))
        k = min(max(1, k), len(base_symptoms))
        chosen = random.sample(base_symptoms, k)
        # occasionally add a random symptom from other diseases for realism
        if random.random() < 0.15:
            other = random.choice(disease_names)
            extra = random.choice(diseases[other])
            if extra not in chosen:
                chosen.append(extra)
        rows.append({"disease": d, "symptoms": ",".join(chosen)})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Synthetic dataset saved to {path} with {len(df)} rows.")
    return df

# -------------------------
# Load dataset
# -------------------------
if Path(DATA_FILE).exists():
    print(f"Loading dataset from {DATA_FILE} ...")
    df = pd.read_csv(DATA_FILE)
    # Check expected columns
    if 'disease' not in df.columns or 'symptoms' not in df.columns:
        raise ValueError("CSV must have 'disease' and 'symptoms' columns (symptoms comma-separated).")
else:
    print(f"No dataset found at {DATA_FILE}. Creating synthetic dataset...")
    df = create_synthetic_dataset(DATA_FILE, n_samples=900)

# Basic EDA prints
print("\nDataset preview:")
print(df.head())
print("\nNumber of records:", len(df))
print("Sample disease distribution:")
print(df['disease'].value_counts())

# -------------------------
# Preprocessing: build symptom vocabulary and one-hot encode
# -------------------------
def preprocess_symptom_dataframe(df):
    # Ensure symptoms column is string
    df['symptoms'] = df['symptoms'].fillna("").astype(str)
    # Normalize: lowercase and strip
    df['symptoms'] = df['symptoms'].apply(lambda s: ",".join([t.strip().lower() for t in s.split(",") if t.strip() != ""]))
    # Build set of all symptoms
    all_symptoms = set()
    for s in df['symptoms']:
        parts = [p for p in s.split(",") if p]
        all_symptoms.update(parts)
    all_symptoms = sorted(list(all_symptoms))
    print(f"\nFound {len(all_symptoms)} unique symptoms.")
    # One-hot encode symptoms
    X = []
    for s in df['symptoms']:
        parts = set([p for p in s.split(",") if p])
        row = [1 if symptom in parts else 0 for symptom in all_symptoms]
        X.append(row)
    X = np.array(X, dtype=int)
    return X, all_symptoms

X, features = preprocess_symptom_dataframe(df)
y_raw = df['disease'].values

# Save features list
with open(FEATURES_FILE, "w") as f:
    json.dump(features, f)
print(f"Saved features (symptom list) to {FEATURES_FILE}.")

# Label encode disease
le = LabelEncoder()
y = le.fit_transform(y_raw)
joblib.dump(le, LABEL_ENCODER_FILE)
print(f"Saved LabelEncoder to {LABEL_ENCODER_FILE}.")

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# -------------------------
# Model training
# -------------------------
clf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
print("\nTraining RandomForestClassifier ...")
clf.fit(X_train, y_train)
joblib.dump(clf, MODEL_FILE)
print(f"Saved trained model to {MODEL_FILE}.")

# Cross-validation score
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
print(f"\nCross-validation accuracy scores: {cv_scores}")
print(f"CV mean accuracy: {cv_scores.mean():.4f}")

# -------------------------
# Evaluation on test set
# -------------------------
y_pred = clf.predict(X_test)
print("\nTest set evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix shape:", cm.shape)

# -------------------------
# Prepare precautions mapping (generic if not provided)
# -------------------------
precautions = {}
# If user provided a 'precautions.json' next to dataset, load it
if Path(PRECAUTIONS_FILE).exists():
    try:
        with open(PRECAUTIONS_FILE, "r") as f:
            precautions = json.load(f)
        print(f"Loaded existing {PRECAUTIONS_FILE}.")
    except Exception:
        precautions = {}

# Fill missing precautions with a default helpful message
for disease in le.classes_:
    if disease not in precautions:
        precautions[disease] = (
            f"{disease}: Rest, stay hydrated, monitor symptoms. "
            "Seek medical attention if symptoms worsen or high fever persists."
        )

with open(PRECAUTIONS_FILE, "w") as f:
    json.dump(precautions, f, indent=2)
print(f"Saved precautions mapping to {PRECAUTIONS_FILE}.")

print("\nTraining complete. Files generated:")
print(" -", MODEL_FILE)
print(" -", LABEL_ENCODER_FILE)
print(" -", FEATURES_FILE)
print(" -", PRECAUTIONS_FILE)
print("\nYou can now run the Streamlit app: `streamlit run app.py`")
