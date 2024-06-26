# -*- coding: utf-8 -*-
"""assignment.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pm6e8k0NlP4dXRDwlh2BY3vqfR0nz0Q3

phase -1 import data and define
"""

import pandas as pd
import numpy as np

dataset_path = "vmCloud_data.csv"
Cloud_data = pd.read_csv(dataset_path)

print(Cloud_data.head())

# Display the columns in the dataset
print(Cloud_data.columns)

"""phase - 2 * preprocessing the data and normalizing the values and cleaning the data

"""

from sklearn.preprocessing import MinMaxScaler

Cloud_data.fillna(Cloud_data.mean(numeric_only=True), inplace=True)

# missing values and stuff
for column in ['task_type', 'task_priority', 'task_status']:
    Cloud_data[column] = Cloud_data[column].fillna(Cloud_data[column].mode()[0])

columns_to_normalize = ['cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption', 'num_executed_instructions', 'execution_time', 'energy_efficiency']

#using the min max scaler
scaler = MinMaxScaler()

Cloud_data[columns_to_normalize] = scaler.fit_transform(Cloud_data[columns_to_normalize])

#check
print(Cloud_data.head())

#check
# 1. Check for Missing Values
print("Missing values per column:")
print(Cloud_data.isnull().sum())

# 2. Check Data Types
print("\nData types of each column:")
print(Cloud_data.dtypes)

# 3. Check Normalization
print("\nMin and Max values for normalized columns:")
for column in ['cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption', 'num_executed_instructions', 'execution_time', 'energy_efficiency']:
    print(f"{column}: Min = {Cloud_data[column].min()}, Max = {Cloud_data[column].max()}")

# 4. Visual Inspection
# Display the first few rows of the DataFrame
print("\nFirst few rows for visual inspection:")
print(Cloud_data.head())

"""Phase - 3 feature engineering - ectracing relevent features from the data"""

#coveting col to datetime
Cloud_data['timestamp'] = pd.to_datetime(Cloud_data['timestamp'])

Cloud_data['hour_of_day'] = Cloud_data['timestamp'].dt.hour
Cloud_data['day_of_week'] = Cloud_data['timestamp'].dt.dayofweek  # Monday=0, Sunday=6

#historical ussage patterns
Cloud_data['cpu_usage_7d_avg'] = Cloud_data['cpu_usage'].rolling(window=7, min_periods=1).mean()

Cloud_data['memory_usage_7d_avg'] = Cloud_data['memory_usage'].rolling(window=7, min_periods=1).mean()

Cloud_data.sort_values('timestamp', inplace=True)

#check
print(Cloud_data.head())

#checking whether historical ussage patterns are calculated or not
#Check for NaN Values
print("NaN values in 'cpu_usage_7d_avg':", Cloud_data['cpu_usage_7d_avg'].isnull().sum())
print("NaN values in 'memory_usage_7d_avg':", Cloud_data['memory_usage_7d_avg'].isnull().sum())

# Visual Inspection
print("\nFirst few rows for visual inspection:")
print(Cloud_data.head())
print("\nLast few rows for visual inspection:")
print(Cloud_data.tail())

# Plotting the calculations
import matplotlib.pyplot as plt

# Plotting CPU Usage and its 7-day rolling avg
plt.figure(figsize=(10, 6))
plt.plot(Cloud_data['timestamp'], Cloud_data['cpu_usage'], label='CPU Usage')
plt.plot(Cloud_data['timestamp'], Cloud_data['cpu_usage_7d_avg'], label='7-Day Rolling Avg of CPU Usage', linestyle='--')
plt.xlabel('Timestamp')
plt.ylabel('CPU Usage')
plt.title('CPU Usage and 7-Day Rolling Average')
plt.legend()
plt.show()

#plotting the memory ussage and its 7-day rolling avg
plt.figure(figsize=(10, 6))
plt.plot(Cloud_data['timestamp'], Cloud_data['memory_usage'], label='Memory Usage')
plt.plot(Cloud_data['timestamp'], Cloud_data['memory_usage_7d_avg'], label='7-Day Rolling Avg of Memory Usage', linestyle='--')
plt.xlabel('Timestamp')
plt.ylabel('Memory Usage')
plt.title('Memory Usage and 7-Day Rolling Average')
plt.legend()
plt.show()

"""phase -4 getting the features and preparing to split the data"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Encode categorical variables
label_encoder = LabelEncoder()
Cloud_data['task_type_encoded'] = label_encoder.fit_transform(Cloud_data['task_type'])
Cloud_data['task_priority_encoded'] = label_encoder.fit_transform(Cloud_data['task_priority'])
Cloud_data['task_status_encoded'] = label_encoder.fit_transform(Cloud_data['task_status'])

# Drop original categorical columns and 'timestamp'
Cloud_data_processed = Cloud_data.drop(['timestamp', 'task_type', 'task_priority', 'task_status'], axis=1)

# Features (excluding target variables and 'vm_id' if it's not used as a feature)
X = Cloud_data_processed.drop(['cpu_usage', 'memory_usage', 'network_traffic', 'vm_id'], axis=1)

# Targets
y = Cloud_data_processed[['cpu_usage', 'memory_usage', 'network_traffic']]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Checking the shape of the splits
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

"""phase -5 model selection and training"""

import lightgbm as lgb

# Prepare the dataset for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters for the model
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Define early stopping callback
early_stopping = lgb.early_stopping(stopping_rounds=10)

# Train the model with early stopping callback
gbm = lgb.train(params,
                train_data,
                num_boost_round=200,
                valid_sets=test_data,
                callbacks=[early_stopping])


# Predict on test set
y_pred_gbm = gbm.predict(X_test, num_iteration=gbm.best_iteration)

print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers.legacy import Adam

# Assuming X_train is initially shaped as [samples, features]
X_train_array = X_train.to_numpy()

# Reshape X_train_array to have three dimensions: [samples, time steps, features]
X_train_reshaped = X_train_array.reshape((X_train_array.shape[0], 1, X_train_array.shape[1]))

# Use X_train_reshaped for the model
model = Sequential([
    GRU(50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(1)  # Assuming y_train is 1D, for a single-output task
])

model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Use X_train_reshaped in model.fit()
history = model.fit(X_train_reshaped, y_train, epochs=10, validation_split=0.2, batch_size=64, verbose=1)

# Print the final training and validation loss
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"Final Training Loss: {final_train_loss}")
print(f"Final Validation Loss: {final_val_loss}")

import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Assuming X_test and y_test are your testing data and labels
X_test_reshaped = X_test.to_numpy().reshape((X_test.shape[0], 1, X_test.shape[1]))

# Evaluate the model on the test data
test_loss = model.evaluate(X_test_reshaped, y_test, verbose=1)

print(f"Test MSE: {test_loss}")

"""phase - 6"""

from sklearn.preprocessing import MinMaxScaler

new_data = pd.read_csv('vmCloud_data.csv')

new_data.fillna(new_data.mean(numeric_only=True), inplace=True)

# missing values and stuff
for column in ['task_type', 'task_priority', 'task_status']:
    new_data[column] = new_data[column].fillna(new_data[column].mode()[0])

columns_to_normalize = ['cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption', 'num_executed_instructions', 'execution_time', 'energy_efficiency']

#using the min max scaler
scaler = MinMaxScaler()

new_data[columns_to_normalize] = scaler.fit_transform(new_data[columns_to_normalize])

#check
print(new_data.head())

#check
# 1. Check for Missing Values
print("Missing values per column:")
print(new_data.isnull().sum())

# 2. Check Data Types
print("\nData types of each column:")
print(new_data.dtypes)

# 3. Check Normalization
print("\nMin and Max values for normalized columns:")
for column in ['cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption', 'num_executed_instructions', 'execution_time', 'energy_efficiency']:
    print(f"{column}: Min = {new_data[column].min()}, Max = {new_data[column].max()}")

# 4. Visual Inspection
# Display the first few rows of the DataFrame
print("\nFirst few rows for visual inspection:")
print(new_data.head())

#coveting col to datetime
new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])

new_data['hour_of_day'] = new_data['timestamp'].dt.hour
new_data['day_of_week'] = new_data['timestamp'].dt.dayofweek

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Encode categorical variables
label_encoder = LabelEncoder()
new_data['task_type_encoded'] = label_encoder.fit_transform(new_data['task_type'])
new_data['task_priority_encoded'] = label_encoder.fit_transform(new_data['task_priority'])
new_data['task_status_encoded'] = label_encoder.fit_transform(new_data['task_status'])

# Drop original categorical columns and 'timestamp'
new_data_processed = new_data.drop(['timestamp', 'task_type', 'task_priority', 'task_status'], axis=1)

# Features (excluding target variables and 'vm_id' if it's not used as a feature)
X = new_data_processed.drop(['cpu_usage', 'memory_usage', 'network_traffic', 'vm_id'], axis=1)

# Targets
y = new_data_processed[['cpu_usage', 'memory_usage', 'network_traffic']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Checking the shape of the splits
print("Training set shape:", X_train.shape, y_train.shape)

# Take only the first 10 rows of X
X_subset = X.iloc[:10]

# Convert X_subset to a numpy array and then reshape it to have 3 dimensions: [samples, time steps, features]
X_reshaped_subset = X_subset.to_numpy().reshape((X_subset.shape[0], 1, X_subset.shape[1]))

# Now, use the reshaped X_subset for prediction
predictions_subset = model.predict(X_reshaped_subset)

# Use predictions as needed
print(predictions_subset)

"""phase - 9 Resource Allocation Optimization"""

# Print the shape of predictions_subset
print("Shape of predictions_subset:", predictions_subset.shape)

# Print the first few rows of predictions_subset
print("First few rows of predictions_subset:", predictions_subset[:5])

import numpy as np
from scipy.optimize import minimize

# Example: predictions_subset contains predicted CPU utilization rates for 10 tasks
# Convert predictions to average utilization per task
avg_utilization_per_task = np.mean(predictions_subset, axis=1)

# Assuming the sum of avg_utilization_per_task represents the total demand
total_demand = sum(avg_utilization_per_task)

def objective(x):
    # Minimize the total allocated resources
    return sum(x) / len(x)  # Average allocation per task

def constraint(x):
    # Ensure total allocation meets or exceeds total demand
    return sum(x) - total_demand

# Adjust initial guess if necessary
x0 = np.full(len(avg_utilization_per_task), 0.1)  # Start with a small allocation per task

# Solve the optimization problem with the adjusted constraint
solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints={'type': 'ineq', 'fun': constraint})

print("Adjusted Optimized Resource Allocation:", solution.x)

