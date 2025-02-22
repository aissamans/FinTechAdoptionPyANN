# -*- coding: utf-8 -*-
"""
Python script for analyzing fintech adoption determinants using Artificial Neural Networks (ANN) with 10-fold cross-validation.

Author: Aissam Mansouri
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# TensorFlow / Keras
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ============================================================
# 1. Data Loading and Initial Setup
# ============================================================

# Replace "YOUR_DATA_FILE.csv" with your actual path or filename
data = pd.read_csv("YOUR_DATA_FILE.csv")

# Create composite variables by averaging related items
data_composite = pd.DataFrame()
data_composite["PE"] = data[[col for col in data.columns if col.startswith("PE")]].mean(axis=1)
data_composite["SI"] = data[[col for col in data.columns if col.startswith("SI")]].mean(axis=1)
data_composite["EE"] = data[[col for col in data.columns if col.startswith("EE")]].mean(axis=1)
data_composite["FC"] = data[[col for col in data.columns if col.startswith("FC")]].mean(axis=1)
data_composite["IT"] = data[[col for col in data.columns if col.startswith("IT")]].mean(axis=1)
data_composite["Openness"] = data[[col for col in data.columns if col.startswith("Openness")]].mean(axis=1)
data_composite["Conscientiousness"] = data[[col for col in data.columns if col.startswith("Conscientiousness")]].mean(axis=1)
data_composite["Extraversion"] = data[[col for col in data.columns if col.startswith("Extraversion")]].mean(axis=1)
data_composite["Agreeableness"] = data[[col for col in data.columns if col.startswith("Agreeableness")]].mean(axis=1)
data_composite["Neuroticism"] = data[[col for col in data.columns if col.startswith("Neuroticism")]].mean(axis=1)

print("Data shape:", data_composite.shape)
print(data_composite.head())

# ============================================================
# 2. ANN Model: FullANN (All Constructs)
#    10-Fold Cross-Validation
# ============================================================

# Separate features (X_full) and target (y_full)
X_full = data_composite.drop(columns=["IT"])
y_full = data_composite["IT"]

# Scale X and y
scaler_X_full = MinMaxScaler()
X_full_scaled = scaler_X_full.fit_transform(X_full)

scaler_y_full = MinMaxScaler()
y_full_scaled = scaler_y_full.fit_transform(y_full.values.reshape(-1, 1))

# Define the ANN architecture
FullANN = Sequential([
    Dense(10, activation="relu", input_shape=(X_full_scaled.shape[1],)),
    Dense(10, activation="relu"),
    Dense(1, activation="linear")
])

# K-fold setup
kf = KFold(n_splits=10, shuffle=True, random_state=70)

rmse_train_original_full = []
rmse_val_original_full = []
r2_train_full = []
r2_val_full = []
adjusted_r2_train_full = []
adjusted_r2_val_full = []

# Cross-validation loop
for train_index, val_index in kf.split(X_full_scaled):
    X_train, X_val = X_full_scaled[train_index], X_full_scaled[val_index]
    y_train, y_val = y_full_scaled[train_index], y_full_scaled[val_index]

    # Clone model to re-initialize for each fold
    model = clone_model(FullANN)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    # Train the model (epochs and batch size may be tuned)
    model.fit(X_train, y_train, epochs=200, batch_size=64, verbose=0)

    # Inverse-scale the targets for training and validation
    y_train_original = scaler_y_full.inverse_transform(y_train)
    y_val_original = scaler_y_full.inverse_transform(y_val)

    # Predict on training and validation sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    y_train_pred_original = scaler_y_full.inverse_transform(y_train_pred)
    y_val_pred_original = scaler_y_full.inverse_transform(y_val_pred)

    # Compute R^2 and adjusted R^2
    r2_train = r2_score(y_train_original, y_train_pred_original)
    r2_val = r2_score(y_val_original, y_val_pred_original)

    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    p = X_train.shape[1]

    # Adjusted R^2
    adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p - 1)
    adj_r2_val = 1 - (1 - r2_val) * (n_val - 1) / (n_val - p - 1)

    # Compute RMSE
    rmse_train = np.sqrt(np.mean((y_train_original - y_train_pred_original) ** 2))
    rmse_val = np.sqrt(np.mean((y_val_original - y_val_pred_original) ** 2))

    # Store metrics
    r2_train_full.append(r2_train)
    r2_val_full.append(r2_val)
    adjusted_r2_train_full.append(adj_r2_train)
    adjusted_r2_val_full.append(adj_r2_val)
    rmse_train_original_full.append(rmse_train)
    rmse_val_original_full.append(rmse_val)

# Compile results into a DataFrame
scores_full = pd.DataFrame({
    "RMSE_Training": rmse_train_original_full,
    "RMSE_Testing": rmse_val_original_full,
    "R^2_train": r2_train_full,
    "R^2_val": r2_val_full,
    "Adjusted_R^2_train": adjusted_r2_train_full,
    "Adjusted_R^2_val": adjusted_r2_val_full
})

# Replace "YOUR_OUTPUT_FILE.csv" with your desired output file path
scores_full.to_csv("YOUR_OUTPUT_FILE_FullANN.csv", index=False)
print("FullANN CV results saved to YOUR_OUTPUT_FILE_FullANN.csv")

# ============================================================
# 3. ANN Model: SEMANN (Significant Variables from SEM)
#    10-Fold Cross-Validation
# ============================================================

# Suppose from your SEM analysis, only PE, FC, and SI are significant
# Modify this list as needed
significant_features = ["PE", "FC", "SI"]

semdata_composite = pd.DataFrame()
semdata_composite["PE"] = data[[col for col in data.columns if col.startswith("PE")]].mean(axis=1)
semdata_composite["SI"] = data[[col for col in data.columns if col.startswith("SI")]].mean(axis=1)
semdata_composite["FC"] = data[[col for col in data.columns if col.startswith("FC")]].mean(axis=1)
semdata_composite["IT"] = data[[col for col in data.columns if col.startswith("IT")]].mean(axis=1)

X_sem = semdata_composite[significant_features]
y_sem = semdata_composite["IT"]

# Scale X and y
scaler_X_sem = MinMaxScaler()
X_sem_scaled = scaler_X_sem.fit_transform(X_sem)

scaler_y_sem = MinMaxScaler()
y_sem_scaled = scaler_y_sem.fit_transform(y_sem.values.reshape(-1, 1))

# Define the SEMANN architecture
SEMANN = Sequential([
    Dense(7, activation="relu", input_shape=(X_sem_scaled.shape[1],)),
    Dense(5, activation="relu"),
    Dense(5, activation="relu"),
    Dense(1, activation="linear")
])

rmse_train_original_sem = []
rmse_val_original_sem = []
r2_train_sem = []
r2_val_sem = []
adjusted_r2_train_sem = []
adjusted_r2_val_sem = []

# Cross-validation loop
kf_sem = KFold(n_splits=10, shuffle=True, random_state=70)
for train_index, val_index in kf_sem.split(X_sem_scaled):
    X_train_sem, X_val_sem = X_sem_scaled[train_index], X_sem_scaled[val_index]
    y_train_sem, y_val_sem = y_sem_scaled[train_index], y_sem_scaled[val_index]

    # Re-initialize the model for each fold
    model_sem = clone_model(SEMANN)
    model_sem.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model (tune epochs/batch_size as needed)
    model_sem.fit(X_train_sem, y_train_sem, epochs=300, batch_size=64, verbose=0)

    # Inverse-scale the y-values
    y_train_original_sem = scaler_y_sem.inverse_transform(y_train_sem)
    y_val_original_sem = scaler_y_sem.inverse_transform(y_val_sem)

    # Predictions
    y_train_pred_sem = model_sem.predict(X_train_sem)
    y_val_pred_sem = model_sem.predict(X_val_sem)

    y_train_pred_original_sem = scaler_y_sem.inverse_transform(y_train_pred_sem)
    y_val_pred_original_sem = scaler_y_sem.inverse_transform(y_val_pred_sem)

    # Compute R^2 and adjusted R^2
    r2_train_s = r2_score(y_train_original_sem, y_train_pred_original_sem)
    r2_val_s = r2_score(y_val_original_sem, y_val_pred_original_sem)

    n_train_s = X_train_sem.shape[0]
    n_val_s = X_val_sem.shape[0]
    p_s = X_val_sem.shape[1]

    adj_r2_train_s = 1 - (1 - r2_train_s) * (n_train_s - 1) / (n_train_s - p_s - 1)
    adj_r2_val_s = 1 - (1 - r2_val_s) * (n_val_s - 1) / (n_val_s - p_s - 1)

    # Compute RMSE
    rmse_train_s = np.sqrt(np.mean((y_train_original_sem - y_train_pred_original_sem) ** 2))
    rmse_val_s = np.sqrt(np.mean((y_val_original_sem - y_val_pred_original_sem) ** 2))

    # Store metrics
    r2_train_sem.append(r2_train_s)
    r2_val_sem.append(r2_val_s)
    adjusted_r2_train_sem.append(adj_r2_train_s)
    adjusted_r2_val_sem.append(adj_r2_val_s)
    rmse_train_original_sem.append(rmse_train_s)
    rmse_val_original_sem.append(rmse_val_s)

# Compile results into a DataFrame
scores_sem = pd.DataFrame({
    "RMSE_Training": rmse_train_original_sem,
    "RMSE_Testing": rmse_val_original_sem,
    "R^2_train": r2_train_sem,
    "R^2_val": r2_val_sem,
    "Adjusted_R^2_train": adjusted_r2_train_sem,
    "Adjusted_R^2_val": adjusted_r2_val_sem
})

# Replace "YOUR_OUTPUT_FILE.csv" with your desired output file path
scores_sem.to_csv("YOUR_OUTPUT_FILE_SEMANN.csv", index=False)
print("SEMANN CV results saved to YOUR_OUTPUT_FILE_SEMANN.csv")
