!pip install catboost -q

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from google.colab import files
import warnings
warnings.filterwarnings("ignore")

print(" Upload train.csv, test.csv and sample_submission.csv")
uploaded = files.upload()

train_key = [k for k in uploaded.keys() if "train" in k.lower()][0]
test_key  = [k for k in uploaded.keys() if "test"  in k.lower()][0]

with open("train.csv", "wb") as f:
    f.write(uploaded[train_key])

with open("test.csv", "wb") as f:
    f.write(uploaded[test_key])

print("Train →", train_key)
print("Test  →", test_key)

TRAIN_FILE = "train.csv"
TEST_FILE  = "test.csv"
SUBMISSION_FILE = "submission_catboost_cv_fixed.csv"

SEED = 42
N_FOLDS = 5

# =========================================================
# PREPROCESSING FUNCTION (FULLY SAFE)
# =========================================================

def preprocess_data(train_path, test_path):
    print("\nLoading data...")
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    print("Train shape:", train.shape)
    print("Test shape :", test.shape)

    print("\nBefore cleaning, spend_category unique values:")
    print(train["spend_category"].value_counts(dropna=False))

    # ---- DROP NA TARGET, NO CAST BEFORE THIS ----
    mask_valid = ~train["spend_category"].isna()
    dropped = len(train) - mask_valid.sum()
    train = train.loc[mask_valid].copy()
    print(f"Dropped {dropped} rows with NaN spend_category.")

    # Safe now — convert to int
    train["spend_category"] = train["spend_category"].astype(int)

    # Dummy label for test
    test["spend_category"] = -1

    # Concat for uniform processing
    df = pd.concat([train, test], axis=0).reset_index(drop=True)

    # -------------------------------
    # NUMERIC FIXES (NO CASTING YET)
    # -------------------------------

    df["num_females"] = df["num_females"].fillna(0)
    df["num_males"]   = df["num_males"].fillna(0)
    df["total_people"] = df["num_females"] + df["num_males"]

    df["mainland_stay_nights"] = df["mainland_stay_nights"].fillna(0)
    df["island_stay_nights"]   = df["island_stay_nights"].fillna(0)
    df["total_nights"] = df["mainland_stay_nights"] + df["island_stay_nights"]

    df["nights_per_person"] = df["total_nights"] / df["total_people"].replace(0, 1)

    df["log_total_nights"]  = np.log1p(df["total_nights"])
    df["log_total_people"]  = np.log1p(df["total_people"])

    # -------------------------------
    # SERVICE FEATURES
    # -------------------------------

    binary_cols = [
        "intl_transport_included",
        "accomodation_included",
        "food_included",
        "domestic_transport_included",
        "sightseeing_included",
        "guide_included",
        "insurance_included",
    ]

    for col in binary_cols:
        df[col] = df[col].fillna("No").astype(str)

    temp = df[binary_cols].replace({"Yes": 1, "No": 0, "yes": 1, "no": 0})
    df["services_count"] = temp.sum(axis=1)
    df["has_any_service"] = (df["services_count"] > 0).astype(int)

    # -------------------------------
    # SAFE ORDINAL ENCODING
    # -------------------------------

    days_map = {"0-7":0, "8-14":1, "15-30":2, "31-60":3, "61-90":4, "90+":5}
    trip_map = {"1-3":0, "4-6":1, "7-14":2, "30+":3}

    df["days_booked_before_trip"] = df["days_booked_before_trip"].fillna("Missing")
    df["days_booked_before_trip_ord"] = (
        df["days_booked_before_trip"]
        .map(days_map)
        .fillna(-1)      # <-- ALWAYS FILL BEFORE CAST
        .astype(int)
    )

    df["total_trip_days"] = df["total_trip_days"].fillna("Missing")
    df["total_trip_days_ord"] = (
        df["total_trip_days"]
        .map(trip_map)
        .fillna(-1)
        .astype(int)
    )

    # -------------------------------
    # FIRST VISIT / REQUIREMENTS
    # -------------------------------

    df["is_first_visit"] = df["is_first_visit"].fillna("No").astype(str)
    df["has_special_requirements"] = df["has_special_requirements"].fillna("none")

    # -------------------------------
    # COUNTRY FEATURES
    # -------------------------------

    country_counts = df["country"].value_counts()
    df["country_freq"] = df["country"].map(country_counts).fillna(0).astype(int)
    df["country_freq_log"] = np.log1p(df["country_freq"])
    df["country_rare"] = (df["country_freq"] <= df["country_freq"].quantile(0.25)).astype(int)

    # -------------------------------
    # CATEGORICAL SETUP
    # -------------------------------

    categorical_cols = [
        "country","age_group","travel_companions","main_activity","visit_purpose",
        "tour_type","info_source","arrival_weather",
        "days_booked_before_trip","total_trip_days",
        "has_special_requirements","is_first_visit"
    ] + binary_cols

    for col in categorical_cols:
        df[col] = df[col].fillna("Missing").astype(str)

    # -------------------------------
    # FINAL SPLIT
    # -------------------------------

    train_proc = df[df["spend_category"] != -1].copy()
    test_proc  = df[df["spend_category"] == -1].copy()

    X = train_proc.drop(columns=["trip_id","spend_category"])
    y = train_proc["spend_category"]
    X_test = test_proc.drop(columns=["trip_id","spend_category"])

    test_ids = test_proc["trip_id"]

    print("\nFinal X shape:", X.shape)
    print("Final X_test shape:", X_test.shape)
    print("Categorical feature count:", len(categorical_cols))

    return X, y, X_test, test_ids, categorical_cols

# =========================================================
# RUN PREPROCESS
# =========================================================
X, y, X_test, test_ids, cat_features = preprocess_data("train.csv", "test.csv")

# =========================================================
# CATBOOST CV
# =========================================================

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

test_probs_sum = np.zeros((X_test.shape[0], 3))
f1_scores = []

print("\n Starting CatBoost K-Fold...")

for fold, (tr, va) in enumerate(skf.split(X, y)):
    print(f"\n===== Fold {fold+1}/{N_FOLDS} =====")

    model = CatBoostClassifier(
        iterations=1600,
        learning_rate=0.035,
        depth=6,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        auto_class_weights="Balanced",
        l2_leaf_reg=4.0,
        random_seed=SEED,
        cat_features=cat_features,
        verbose=False,
        early_stopping_rounds=100,
    )

    model.fit(
        X.iloc[tr], y.iloc[tr],
        eval_set=(X.iloc[va], y.iloc[va]),
        use_best_model=True
    )

    pred_va = model.predict(X.iloc[va]).astype(int).reshape(-1)
    f1 = f1_score(y.iloc[va], pred_va, average="weighted")
    f1_scores.append(f1)
    print("Fold F1:", f1)

    test_probs_sum += model.predict_proba(X_test)

print("\nAverage F1:", np.mean(f1_scores))

# =========================================================
# SAVE SUBMISSION
# =========================================================

final_preds = np.argmax(test_probs_sum / N_FOLDS, axis=1)
sub = pd.DataFrame({"trip_id": test_ids, "spend_category": final_preds})
sub.to_csv(SUBMISSION_FILE, index=False)

print("Saved:", SUBMISSION_FILE)
print(sub.head())
