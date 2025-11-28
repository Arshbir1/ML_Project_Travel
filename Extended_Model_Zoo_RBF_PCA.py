# ============================================================
# Kaggle: Tourist Travel Spend_Category - EXTENDED MODEL ZOO
# Models: Logistic, RF, GB, XGB, MLP, CatBoost, LightGBM,
#         ExtraTrees, HistGB, RBF-SVM(PCA), Naive Bayes
# ============================================================

import io
import numpy as np
import pandas as pd
from google.colab import files

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Base models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Histogram-based boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# PCA for SVM
from sklearn.decomposition import PCA

# XGBoost / CatBoost / LightGBM
!pip install xgboost catboost lightgbm -q
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# ============================================================
# 1) FILE UPLOAD
# ============================================================

print(" Upload train.csv, test.csv, sample_submission.csv")
uploaded = files.upload()

def pick_file(keyword):
    keyword = keyword.lower()
    for k in uploaded.keys():
        if keyword in k.lower():
            return k
    raise ValueError(f"File containing '{keyword}' not found.")

train_key  = pick_file("train")
test_key   = pick_file("test")
sample_key = pick_file("sample")

train_df = pd.read_csv(io.BytesIO(uploaded[train_key]))
test_df  = pd.read_csv(io.BytesIO(uploaded[test_key]))
sample_df = pd.read_csv(io.BytesIO(uploaded[sample_key]))

print("\nTrain shape:", train_df.shape)
print("Test shape :", test_df.shape)
print(train_df.head())

# ============================================================
# 2) BASIC SETUP
# ============================================================
ID_COL = "trip_id"
TARGET = "spend_category"

# Drop rows with missing target
train_df = train_df[train_df[TARGET].notna()].copy()
train_df[TARGET] = train_df[TARGET].astype(int)

X = train_df.drop([ID_COL, TARGET], axis=1)
y = train_df[TARGET]

test_ids = test_df[ID_COL]
X_test = test_df.drop([ID_COL], axis=1)

# Determine column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# ============================================================
# 3) PREPROCESSOR (Impute + OneHot)
# ============================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# ============================================================
# CREATE PREPROCESSED DATA (fit once)
# ============================================================
X_enc = preprocess.fit_transform(X)
X_test_enc = preprocess.transform(X_test)

print("\nEncoded shapes:")
print("X_enc:", X_enc.shape)
print("X_test_enc:", X_test_enc.shape)

# ============================================================
# TRAIN/VALID SPLIT
# ============================================================
X_tr, X_val, y_tr, y_val = train_test_split(
    X_enc, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain/Valid:")
print(X_tr.shape, X_val.shape)

# ============================================================
# MODEL ZOO
# ============================================================

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, n_jobs=-1),
    "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "extra_trees": ExtraTreesClassifier(n_estimators=400, n_jobs=-1, random_state=42),
    "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=None, random_state=42),

    # Boosting big guns
    "xgboost": XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42
    ),
    "catboost": CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        verbose=False
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=40,
        random_state=42
    ),

    # Neural Network
    "neural_net": MLPClassifier(hidden_layer_sizes=(256,128), max_iter=40, random_state=42),

    # Naive Bayes
    "naive_bayes": GaussianNB(),

    # RBF SVM with PCA compression
    "svm_rbf_pca": Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=60)),
        ("svm", SVC(kernel="rbf", C=2.0, gamma="scale"))
    ])
}

# ============================================================
# VALIDATION LOOP
# ============================================================

print("\n=========== VALIDATION ===========")
val_scores = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_tr, y_tr)
    pred = model.predict(X_val)

    acc = accuracy_score(y_val, pred)
    f1  = f1_score(y_val, pred, average="macro")
    val_scores[name] = (acc, f1)

    print(f"{name} → ACC={acc:.4f}  F1={f1:.4f}")

print("\n====== SUMMARY OF MODELS ======")
for k, (acc, f1) in val_scores.items():
    print(f"{k:25s}  ACC={acc:.4f}  F1={f1:.4f}")

# ============================================================
# TRAIN ON FULL DATA & SAVE SUBMISSIONS
# ============================================================

from google.colab import files as colab_files

print("\n=========== FULL TRAIN + SUBMISSIONS ===========")

for name, model in models.items():
    print(f"\nFitting FULL model: {name}")
    model.fit(X_enc, y)

    print("Predicting on test...")
    test_pred = model.predict(X_test_enc)

    # Build submission
    sub = pd.DataFrame({
        "trip_id": test_ids,
        "spend_category": test_pred.astype(int)
    })

    out_name = f"submission_{name}.csv"
    sub.to_csv(out_name, index=False)
    print("Saved:", out_name)

    colab_files.download(out_name)

print("\n DONE — all advanced model submissions generated!")
