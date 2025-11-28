import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# ======================================================
# 1. Load data
# ======================================================

# Change paths as needed (e.g. "train(1).csv" / "test(1).csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target_col = "spend_category"
id_col = "trip_id"

# ======================================================
# 2. Drop rows with missing target
# ======================================================
train = train.dropna(subset=[target_col]).copy()
train[target_col] = train[target_col].astype(int)

# ======================================================
# 3. Cleaning + Feature Engineering
# ======================================================

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- strip trailing commas / dots and lowercase for some cols ---
    strip_cols = ["arrival_weather", "has_special_requirements"]
    for col in strip_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip(" ,.")
            )
            df[col].replace({"nan": np.nan}, inplace=True)

    # --- normalize Yes/No style columns ---
    yn_cols = [
        "is_first_visit", "intl_transport_included", "accomodation_included",
        "food_included", "domestic_transport_included", "sightseeing_included",
        "guide_included", "insurance_included"
    ]
    for col in yn_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            df[col].replace({"yes": "Yes", "no": "No", "nan": np.nan}, inplace=True)

    # --- feature: total group size ---
    if {"num_females", "num_males"}.issubset(df.columns):
        df["total_group_size"] = df["num_females"].fillna(0) + df["num_males"].fillna(0)

    # --- feature: is_family_trip ---
    if "travel_companions" in df.columns:
        family_terms = ["spouse", "children", "family"]

        def is_family(x):
            x = str(x).lower()
            return int(any(term in x for term in family_terms))

        df["is_family_trip"] = df["travel_companions"].apply(is_family)

    # --- feature: total_stay_nights & clip outliers ---
    if {"mainland_stay_nights", "island_stay_nights"}.issubset(df.columns):
        df["total_stay_nights"] = (
            df["mainland_stay_nights"].fillna(0) +
            df["island_stay_nights"].fillna(0)
        )
        df["total_stay_nights"] = df["total_stay_nights"].clip(0, 60)

    # --- feature: island_ratio ---
    if "total_stay_nights" in df.columns and "island_stay_nights" in df.columns:
        tsn = df["total_stay_nights"].replace(0, np.nan)
        df["island_ratio"] = df["island_stay_nights"] / tsn
        df["island_ratio"] = df["island_ratio"].fillna(0)

    # --- ordinal mapping: days_booked_before_trip ---
    days_map = {
        "0-7": 1,
        "8-14": 2,
        "15-30": 3,
        "31-60": 4,
        "61-90": 5,
        "90+": 6,
    }
    if "days_booked_before_trip" in df.columns:
        df["days_booked_before_trip_ord"] = df["days_booked_before_trip"].map(days_map)

    # --- ordinal mapping: total_trip_days ---
    trip_map = {
        "1-3": 1,
        "1-6": 2,   # we saw 1-6 in sample
        "4-6": 3,
        "7-14": 4,
        "15-30": 5,
        "30+": 6,
    }
    if "total_trip_days" in df.columns:
        df["total_trip_days_ord"] = df["total_trip_days"].map(trip_map)

    # --- binary first visit ---
    if "is_first_visit" in df.columns:
        df["is_first_visit_bin"] = df["is_first_visit"].map({"Yes": 1, "No": 0})

    # Ensure engineered numerics are float
    num_cols_engineered = [
        c for c in [
            "total_group_size", "is_family_trip", "total_stay_nights",
            "island_ratio", "days_booked_before_trip_ord",
            "total_trip_days_ord", "is_first_visit_bin"
        ]
        if c in df.columns
    ]
    df[num_cols_engineered] = df[num_cols_engineered].astype(float)

    return df


train_fe = clean_and_engineer(train)
test_fe = clean_and_engineer(test)

# ======================================================
# 4. Split X, y and define feature lists
# ======================================================
y = train_fe[target_col]
X = train_fe.drop(columns=[target_col])

feature_cols = [c for c in X.columns if c != id_col]
X_features = X[feature_cols]

numeric_cols = X_features.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Train/validation split for model evaluation
X_train, X_val, y_train, y_val = train_test_split(
    X_features, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================================
# 5. Preprocessor (imputer + scaler + one-hot)
# ======================================================
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# ======================================================
# 6. Class weights for imbalance
# ======================================================
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)

# ======================================================
# 7. SVM with GridSearchCV (hyperparameter tuning)
# ======================================================
svm_pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", SVC(
            kernel="rbf",
            probability=False,
            class_weight=class_weight_dict,
            random_state=42,
        )),
    ]
)

param_grid = {
    "model__C": [1, 3, 5],
    "model__gamma": ["scale", 0.1, 0.01],
}

grid_svm = GridSearchCV(
    svm_pipe,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

print("\n>>> Running GridSearchCV for SVM...")
grid_svm.fit(X_train, y_train)

print("\nBest SVM parameters:", grid_svm.best_params_)
print("Best CV accuracy:", grid_svm.best_score_)

best_svm = grid_svm.best_estimator_

# Validation performance
y_val_pred = best_svm.predict(X_val)
print("\nValidation Accuracy (best SVM):", accuracy_score(y_val, y_val_pred))
print("\nClassification report (best SVM):")
print(classification_report(y_val, y_val_pred))

# ======================================================
# 8. Train final SVM on full data with best params
# ======================================================
print("\n>>> Training final SVM on full training data...")

best_params = grid_svm.best_params_

svm_final = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", SVC(
            kernel="rbf",
            C=best_params["model__C"],
            gamma=best_params["model__gamma"],
            probability=False,
            class_weight=class_weight_dict,
            random_state=42,
        )),
    ]
)

svm_final.fit(X_features, y)

# ======================================================
# 9. Predict on test & create submission (Option A: labels)
# ======================================================
test_features = test_fe[feature_cols]
test_pred = svm_final.predict(test_features).astype(int)

submission = pd.DataFrame({
    "trip_id": test_fe[id_col],
    "category": test_pred
})

submission.to_csv("submission_svm_robust.csv", index=False)
print("\nSaved submission file as: submission_svm_robust.csv")
