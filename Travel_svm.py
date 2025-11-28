import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# --- 1. Data Loading & Configuration ---
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "submission.csv"

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

# Drop training rows with missing target
train = train.dropna(subset=['spend_category'])
train['spend_category'] = train['spend_category'].astype(int)

# Save IDs for submission
test_ids = test['trip_id']
train_y = train['spend_category']

# --- 2. Advanced Feature Engineering ---

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Impute missing numericals with 0 (logical assumption for counts)
        for col in ['num_females', 'num_males', 'mainland_stay_nights', 'island_stay_nights']:
            X[col] = X[col].fillna(0)

        # Derived Features
        X['total_people'] = X['num_females'] + X['num_males']
        X['is_alone'] = (X['total_people'] == 1).astype(int)
        X['total_stay_nights'] = X['mainland_stay_nights'] + X['island_stay_nights']

        # Service Density (Proxy for Package Value)
        binary_cols = [
            'intl_transport_included', 'accomodation_included', 'food_included',
            'domestic_transport_included', 'sightseeing_included',
            'guide_included', 'insurance_included'
        ]
        # robust mapping
        for col in binary_cols:
            X[col] = X[col].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)

        X['services_count'] = X[binary_cols].sum(axis=1)

        # Text Cleaning
        X['arrival_weather'] = X['arrival_weather'].astype(str).str.lower().str.replace(',', '').replace('nan', 'missing')
        X['has_special_requirements'] = X['has_special_requirements'].astype(str).str.lower().str.replace(',', '').replace('nan', 'none')

        # Range to Numeric Conversion (Midpoint Mapping)
        booking_map = {'0-7': 3.5, '8-14': 11, '15-30': 22.5, '31-60': 45.5, '61-90': 75.5, '90+': 100}
        X['days_booked_numeric'] = X['days_booked_before_trip'].map(booking_map).fillna(-1)

        trip_len_map = {'1-6': 3.5, '1-3': 2, '4-6': 5, '7-14': 10.5, '15-30': 22.5, '30+': 40}
        X['trip_len_numeric'] = X['total_trip_days'].map(trip_len_map).fillna(-1)

        return X

# Apply Engineering
fe = FeatureEngineer()
train_eng = fe.transform(train)
test_eng = fe.transform(test)

# Define Feature Groups
numeric_features = [
    'num_females', 'num_males', 'total_people', 'mainland_stay_nights',
    'island_stay_nights', 'total_stay_nights', 'services_count',
    'days_booked_numeric', 'trip_len_numeric', 'is_alone'
]
categorical_features = [
    'country', 'age_group', 'travel_companions', 'main_activity',
    'visit_purpose', 'is_first_visit', 'tour_type', 'info_source',
    'arrival_weather', 'has_special_requirements'
]

# Handle missing in categoricals before encoding
for col in categorical_features:
    train_eng[col] = train_eng[col].fillna('missing').astype(str)
    test_eng[col] = test_eng[col].fillna('missing').astype(str)

X_train = train_eng[numeric_features + categorical_features]
X_test = test_eng[numeric_features + categorical_features]

# --- 3. Pipeline Definitions ---

# Pipeline A: For Linear/NN Models (Needs Scaling + OHE)
# Use TargetEncoder for High Cardinality 'country' to avoid dimensionality explosion
linear_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat_low', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         [c for c in categorical_features if c != 'country']),
        ('cat_high', TargetEncoder(target_type='continuous'), ['country'])
    ]
)

# Pipeline B: For Tree Models (Needs Ordinal Encoding)
tree_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
    ]
)

# --- 4. Model Definitions ---

# 1. Neural Network (MLP)
mlp_pipeline = Pipeline([
    ('preprocessor', linear_preprocessor),
    ('classifier', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
])

# 2. Support Vector Machine (SVM)
svm_pipeline = Pipeline([
    ('preprocessor', linear_preprocessor),
    ('classifier', SVC(probability=True, class_weight='balanced', random_state=42))
])

# 3. Gradient Boosting (Tree)
# Note: Indices for categorical features are offset by numeric features count
cat_indices = list(range(len(numeric_features), len(numeric_features) + len(categorical_features)))
gb_pipeline = Pipeline([
    ('preprocessor', tree_preprocessor),
    ('classifier', HistGradientBoostingClassifier(
        categorical_features=cat_indices,
        learning_rate=0.05, max_iter=500, random_state=42
    ))
])

# --- 5. Ensemble & Training ---

print("Training Voting Ensemble (GBM + MLP + SVM)...")
ensemble = VotingClassifier(
    estimators=[
        ('gb', gb_pipeline),
        ('mlp', mlp_pipeline),
        ('svm', svm_pipeline)
    ],
    voting='soft',
    weights=[3, 1, 1] # Weighted towards GBM (typically best for this data)
)

ensemble.fit(X_train, train_y)

# --- 6. Prediction ---

print("Generating predictions...")
predictions = ensemble.predict(X_test)

submission = pd.DataFrame({
    'trip_id': test_ids,
    'spend_category': predictions
})

submission.to_csv("submission1.csv", index=False)
print(f"Successfully saved submission1.csv")
print(submission.head())
