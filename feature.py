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