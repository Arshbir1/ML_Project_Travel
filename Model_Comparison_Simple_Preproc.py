# ============================================================
# Kaggle: Tourist Travel / spend_category (0,1,2) classification
# Multi-model training + multiple submission files
# Models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, MLP (Neural Net)
# ============================================================

import io
import numpy as np
import pandas as pd

from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# XGBoost
!pip install xgboost -q
from xgboost import XGBClassifier

# -----------------------------
# 1) Upload files
# -----------------------------
print("📤 Upload train.csv, test.csv, sample_submission.csv")
uploaded = files.upload()

def find_key(uploaded_dict, pattern):
    pattern = pattern.lower()
    for k in uploaded_dict.keys():
        if pattern in k.lower():
            return k
    raise ValueError(f"No file containing '{pattern}' found in uploaded files: {list(uploaded_dict.keys())}")

# Find matching filenames, even if Colab renames them like 'train (1).csv'
train_key  = find_key(uploaded, "train")
test_key   = find_key(uploaded, "test")
sample_key = find_key(uploaded, "sample")

print("\nLoaded files:")
print("Train :", train_key)
print("Test  :", test_key)
print("Sample:", sample_key)

# Read dataframes
train_df = pd.read_csv(io.BytesIO(uploaded[train_key]))
test_df  = pd.read_csv(io.BytesIO(uploaded[test_key]))
sample_submission = pd.read_csv(io.BytesIO(uploaded[sample_key]))

print("\nTrain shape:", train_df.shape)
print("Test shape :", test_df.shape)

print("\nTrain columns:", train_df.columns.tolist())

# -----------------------------
# 2) Basic setup
# -----------------------------
ID_COL = "trip_id"
TARGET_COL = "spend_category"

print("\nTarget value counts (including NaNs):")
print(train_df[TARGET_COL].value_counts(dropna=False))

# Drop rows where target is NaN
before_rows = len(train_df)
train_df = train_df.dropna(subset=[TARGET_COL])
after_rows = len(train_df)
print(f"\nDropped {before_rows - after_rows} rows with NaN in '{TARGET_COL}'.")
print("New train shape:", train_df.shape)

# -----------------------------
# 3) Split features & target
# -----------------------------
X = train_df.drop([ID_COL, TARGET_COL], axis=1)
y = train_df[TARGET_COL].astype(int)   # ensure 0/1/2 int labels

test_ids = test_df[ID_COL]
X_test = test_df.drop([ID_COL], axis=1)

# Identify numeric vs categorical columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# -----------------------------
# 4) Preprocessing: Impute + OneHot
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Fit on training, transform both train & test
X_enc = preprocess.fit_transform(X)
X_test_enc = preprocess.transform(X_test)

print("\nEncoded feature shapes:")
print("X_enc     :", X_enc.shape)
print("X_test_enc:", X_test_enc.shape)

# -----------------------------
# 5) Train/validation split
# -----------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X_enc, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain/Valid split:")
print("X_train:", X_train.shape, "X_valid:", X_valid.shape)

# -----------------------------
# 6) Define models
# -----------------------------
models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        multi_class="auto",
        n_jobs=-1
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    ),
    "gradient_boosting": GradientBoostingClassifier(
        random_state=42
    ),
    "xgboost": XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",   # directly outputs class labels 0/1/2
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42
    ),
    "neural_net": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=50,
        random_state=42
    )
}

# -----------------------------
# 7) Validation loop
# -----------------------------
print("\n================ MODEL VALIDATION ================")

val_scores = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred_valid = model.predict(X_valid)

    acc = accuracy_score(y_valid, y_pred_valid)
    f1m = f1_score(y_valid, y_pred_valid, average="macro")

    val_scores[name] = (acc, f1m)
    print(f"{name} → Accuracy: {acc:.4f} | F1(macro): {f1m:.4f}")

print("\nSummary of validation scores:")
for name, (acc, f1m) in val_scores.items():
    print(f"{name:20s}  Acc: {acc:.4f}  F1(macro): {f1m:.4f}")

# -----------------------------
# 8) Train on FULL data & save submissions
# -----------------------------
print("\n================ TRAIN ON FULL DATA & SAVE SUBMISSIONS ================")

from google.colab import files as colab_files

for name, model in models.items():
    print(f"\nFitting {name} on FULL training data...")
    model.fit(X_enc, y)

    print(f"Predicting on test set with {name}...")
    test_pred = model.predict(X_test_enc)   # already 0/1/2 ints for all models

    # Build submission dataframe from scratch (do NOT reuse sample_submission shape)
    sub_df = pd.DataFrame({
        ID_COL: test_ids,
        TARGET_COL: test_pred.astype(int)
    })

    out_name = f"submission_{name}.csv"
    sub_df.to_csv(out_name, index=False)
    print(f"Saved {out_name}  with shape {sub_df.shape}")

    # Trigger download in Colab
    colab_files.download(out_name)

print("\n✅ Done! You now have separate submission CSVs for each model.")
