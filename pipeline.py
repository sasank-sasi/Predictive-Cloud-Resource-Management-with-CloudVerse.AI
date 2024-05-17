import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# Custom transformer for filling missing values
class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Impute missing values:
        - Columns of dtype object are imputed with the most frequent value in the column.
        - Columns of other types are imputed with mean of the column.
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].mode()[0] 
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

# Custom transformer for scaling
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.scaler = MinMaxScaler()
        self.columns = columns

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X, y=None):
        X[self.columns] = self.scaler.transform(X[self.columns])
        return X
    
# Define columns to scale
columns_to_scale = ['cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption', 'num_executed_instructions', 'execution_time', 'energy_efficiency']

# Create the preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[
    ('imputer', DataFrameImputer()),
    ('scaler', CustomScaler(columns=columns_to_scale))
])

dataset_path = "subpart_data.csv"
new = pd.read_csv(dataset_path)

# Assuming `new_data` is your DataFrame with new or incoming data
new_data = preprocessing_pipeline.fit_transform(new)

# Now `new_data_preprocessed` is ready for forecasting
print(new_data.head())

from joblib import load
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Encode categorical variables
label_encoder = LabelEncoder()
new_data['task_type_encoded'] = label_encoder.fit_transform(new_data['task_type'])
new_data['task_priority_encoded'] = label_encoder.fit_transform(new_data['task_priority'])
new_data['task_status_encoded'] = label_encoder.fit_transform(new_data['task_status'])

# Drop original categorical columns and 'timestamp' from the new data
new_data_processed = new_data.drop(['timestamp', 'task_type', 'task_priority', 'task_status'], axis=1)

# Assuming 'vm_id' is not used as a feature, drop it along with the target variables
X_new = new_data_processed.drop(['cpu_usage', 'memory_usage', 'network_traffic', 'vm_id'], axis=1)

# Load the trained models
model_cpu_lightgbm = load('models/lgbm_cpu_usage.joblib')
model_memory_randomforest = load('models/rf_memory_usage.joblib')
model_network_lightgbm = load('models/lgbm_network_traffic.joblib')

# Predict future resource utilization using X_new
cpu_predictions_lightgbm = model_cpu_lightgbm.predict(X_new)
memory_predictions_randomforest = model_memory_randomforest.predict(X_new)
network_predictions_lightgbm = model_network_lightgbm.predict(X_new)

# The predictions are now available in cpu_predictions_lightgbm, memory_predictions_randomforest, and network_predictions_lightgbm
# Further actions can be taken based on these predictions, such as monitoring, scaling, or alerting mechanisms.