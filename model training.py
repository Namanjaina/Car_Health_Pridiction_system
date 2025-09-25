# -*- coding: utf-8 -*-
"""
Training Model - Balanced + Fast Version

1. Loads the upgraded vehicle maintenance dataset.
2. Performs preprocessing and feature engineering.
3. Balances data using 'None' as the normal state.
4. Trains an XGBoost model (fast mode for quick training).
5. Prints accuracy & classification report.
6. Saves model + scaler + encoder + training columns.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle
import warnings

warnings.filterwarnings('ignore')

print("--- TRAINING MODEL (Balanced + Fast Mode) STARTED ---")

# --- Step 1: Load Data ---
try:
    df = pd.read_csv("upgraded_car_data.csv")
    print("Successfully loaded 'upgraded_car_data.csv'.")
except FileNotFoundError:
    print("\nERROR: 'upgraded_car_data.csv' not found.")
    exit()

# --- Step 2: Preprocessing & Feature Engineering ---
df['failure_type'] = df['failure_type'].fillna('None')
df.dropna(inplace=True)
df['failure_type'] = df['failure_type'].astype(str)

# Feature Engineering
df['temp_pressure_ratio'] = df['engine_temp_c'] / (df['oil_pressure_kpa'] + 1e-6)
df['total_brake_wear'] = df['brake_pad_wear_mm_front'] + df['brake_pad_wear_mm_rear']

X = df.drop(columns=['vehicle_id', 'timestamp', 'failure_type'])
y_text = df['failure_type']
le = LabelEncoder()
y_encoded = le.fit_transform(y_text)
training_columns = list(X.columns)

# --- Step 3: Balancing ---
print("Balancing the dataset using 'None' as normal class...")
normal_class_id = le.transform(['None'])[0]
df_to_balance = pd.concat([X, pd.Series(y_encoded, name='failure_type', index=X.index)], axis=1)
df_failures = df_to_balance[df_to_balance.failure_type != normal_class_id]
df_normal = df_to_balance[df_to_balance.failure_type == normal_class_id]

df_normal_upsampled = resample(df_normal,
                             replace=True,
                             n_samples=len(df_failures) // 2,
                             random_state=42)
df_balanced = pd.concat([df_failures, df_normal_upsampled])

print(f"Balanced training dataset size: {len(df_balanced)}")

y_final = df_balanced['failure_type']
X_final = df_balanced.drop('failure_type', axis=1)

# --- Step 4: Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)
print("Scaling complete.")

# --- Step 5: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_final,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y_final)

# --- Step 6: Train Fast XGBoost Model ---
print("\nTraining fast XGBoost model (n_estimators=50, max_depth=4)...")
final_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    n_estimators=50,      # Faster training
    max_depth=4,          # Less complexity, prevents timeout
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
final_model.fit(X_train, y_train)
print("Training complete.")

# --- Step 8: Save Model & Preprocessing Objects ---
print("\nSaving model and preprocessing objects...")

files_to_save = {
    'model.pkl': final_model,
    'scaler.pkl': scaler,
    'encoder.pkl': le,
    'training_columns.pkl': training_columns
}

for filename, obj in files_to_save.items():
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved {filename}")

print("\n--- PROCESS COMPLETE ---")
print("Fast model trained, evaluated, and saved successfully.")
