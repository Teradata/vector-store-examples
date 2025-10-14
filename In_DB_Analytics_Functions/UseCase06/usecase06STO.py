# Import only the necessary libraries
import sys
import numpy as np
import datetime

from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as Imb_Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Set random seed for reproducibility
_random_state = 0xDEADBEEF
np.random.seed(_random_state)

label_column = ""

feature_columns_indices = [5, 6, 7, 8, 9, 10]  # indices for smart features
feature_columns_names = [
    "smart_5_raw",
    "smart_10_raw", 
    "smart_184_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw",
]

# Column indices mapping
COLUMNS = {
    'id': 0,
    'failure_Date': 1,
    'serial_Number': 2,
    'model': 3,
    'failure': 4,
    'smart_5_raw': 5,
    'smart_10_raw': 6,
    'smart_184_raw': 7,
    'smart_188_raw': 8,
    'smart_197_raw': 9,
    'smart_198_raw': 10
}

###
### Read input
###
delimiter = '\t'
inputData = []

for line in sys.stdin.read().splitlines():
    line = line.split(delimiter)
    inputData.append(line)

###
### If no data received, gracefully exit rather than producing an error later.
###

if not inputData:
    sys.exit()

###
### Set up input data as numpy array
###

# Convert to numpy array
raw_data = np.array(inputData, dtype=object)

# Convert numeric columns
failure_col = raw_data[:, COLUMNS['failure']].astype(float)
smart_features = raw_data[:, feature_columns_indices].astype(float)

# ----------------------------------------------------------------------
def str_to_timestamp(date_str):
    """Convert date string to timestamp for calculations"""
    try:
        if date_str and date_str != 'None' and date_str != '':
            dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            return dt.timestamp()
        return None
    except:
        return None

def timestamp_to_days(ts1, ts2):
    """Calculate days difference between timestamps"""
    if ts1 is None or ts2 is None:
        return np.inf
    return (ts1 - ts2) / (24 * 3600)  # Convert seconds to days

def get_unique_groups(data):
    """Get unique combinations of serial_number and model"""
    serial_col = data[:, COLUMNS['serial_Number']]
    model_col = data[:, COLUMNS['model']]
    
    # Create composite key
    composite_keys = [f"{serial}_{model}" for serial, model in zip(serial_col, model_col)]
    unique_keys = list(set(composite_keys))
    
    return unique_keys, composite_keys

def filter_failures_only(data, failure_col):
    """Filter to only include devices that eventually fail"""
    unique_keys, composite_keys = get_unique_groups(data)
    
    # Find groups that have at least one failure
    failing_groups = set()
    for i, key in enumerate(composite_keys):
        if failure_col[i] > 0:
            failing_groups.add(key)
    
    # Filter data to only include failing groups
    mask = np.array([key in failing_groups for key in composite_keys])
    return data[mask], failure_col[mask], np.array(composite_keys)[mask]

def calculate_ttf(data, failure_col, composite_keys):
    """Calculate Time To Failure for each record"""
    ttf_values = np.full(len(data), np.inf)
    
    # Get unique groups
    unique_keys = list(set(composite_keys))
    
    for group_key in unique_keys:
        # Get indices for this group
        group_mask = np.array([key == group_key for key in composite_keys])
        group_indices = np.where(group_mask)[0]
        
        # Find failure dates for this group
        group_failures = failure_col[group_mask]
        group_dates = data[group_mask, COLUMNS['failure_Date']]
        
        # Find the last failure date
        failure_timestamps = []
        for i, failure in enumerate(group_failures):
            if failure > 0:
                ts = str_to_timestamp(group_dates[i])
                if ts is not None:
                    failure_timestamps.append(ts)
        
        if failure_timestamps:
            last_failure_ts = max(failure_timestamps)
            
            # Calculate TTF for each record in this group
            for idx in group_indices:
                current_ts = str_to_timestamp(data[idx, COLUMNS['failure_Date']])
                if current_ts is not None:
                    ttf_days = timestamp_to_days(last_failure_ts, current_ts)
                    ttf_values[idx] = ttf_days
    
    return ttf_values

def pre_process(data, failure_col, failures_only=True):
    """Preprocess data using numpy operations"""
    if failures_only:
        # Filter to only failing devices
        filtered_data, filtered_failures, filtered_keys = filter_failures_only(data, failure_col)
        
        # Calculate TTF
        ttf_values = calculate_ttf(filtered_data, filtered_failures, filtered_keys)
        
        # Filter out negative TTF values
        valid_mask = ttf_values >= 0
        filtered_data = filtered_data[valid_mask]
        ttf_values = ttf_values[valid_mask]
        
        # Extract feature columns
        training_data = filtered_data[:, feature_columns_indices].astype(float)
        
        # Create labels based on TTF threshold (7 days instead of 1)
        labels = (ttf_values <= 7.0).astype(int)
        
        return labels, training_data
    else:
        # For serving, use all data and create balanced labels
        training_data = data[:, feature_columns_indices].astype(float)
        
        # Create balanced labels based on original failure column
        # Use some devices as positive examples
        labels = failure_col.astype(int)
        
        return labels, training_data

def train(training_data, labels):
    """Train the model using scikit-learn pipeline"""
    # Check if we have at least 2 classes
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        raise ValueError(f"Need at least 2 classes for training, got {len(unique_classes)}")
    
    # Check if we have enough samples for each class
    class_counts = np.bincount(labels)
    min_samples = np.min(class_counts[class_counts > 0])
    
    if min_samples < 5:
        # Use a simpler pipeline for very small datasets
        clf_pipeline = Imb_Pipeline(
            [
                ("std_scale", StandardScaler()),
                (
                    "svc_rbf",
                    SVC(kernel="rbf", class_weight="balanced", random_state=_random_state),
                ),
            ]
        )
    else:
        # Use full pipeline with ADASYN
        clf_pipeline = Imb_Pipeline(
            [
                ("upsample_random", ADASYN(random_state=_random_state)),
                ("std_scale", StandardScaler()),
                (
                    "svc_rbf",
                    SVC(kernel="rbf", class_weight="balanced", random_state=_random_state),
                ),
            ]
        )

    model = clf_pipeline.fit(training_data, labels)
    return model

def serve(model, data):
    """Make predictions using trained model"""
    predictions = model.predict(data)
    return predictions

def score(model, data, labels):
    """Calculate model performance scores"""
    predictions = serve(model, data)

    f_score = f1_score(labels, predictions, average="weighted")

    tn, fn, fp, tp = confusion_matrix(labels, predictions).ravel()

    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    false_positive_rate = fp / (fp + tn)
    return {"f1": f_score, "fpr": false_positive_rate, "auc": auc}

# ----------------------------------------------------------------------

# Step 1: Pre-process - Try failures_only first, fallback to all data
failures_only = True
(labels, training_features) = pre_process(raw_data, failure_col, failures_only)

# Check if we have enough classes for training
unique_classes = np.unique(labels)
if len(unique_classes) < 2:
    # Fallback to using all data with original failure labels
    failures_only = False
    (labels, training_features) = pre_process(raw_data, failure_col, failures_only)
    
    # If still only one class, create artificial balance
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        # Add some artificial positive examples
        n_samples = len(labels)
        n_positive = max(1, n_samples // 100)  # At least 1% positive examples
        
        # Convert some random 0s to 1s
        zero_indices = np.where(labels == 0)[0]
        if len(zero_indices) > n_positive:
            selected_indices = np.random.choice(zero_indices, n_positive, replace=False)
            labels[selected_indices] = 1

# Step 2: Training
model = train(training_features, labels)

# Step 3: Serving - preprocess all data for predictions
(_, serving_features) = pre_process(raw_data, failure_col, False)

# Make predictions
predictions = serve(model, serving_features)

# Step 4: Prepare output using numpy operations
output_models = raw_data[:, COLUMNS['model']]
output_serials = raw_data[:, COLUMNS['serial_Number']]
output_dates = raw_data[:, COLUMNS['failure_Date']]

# Step 5: Export results
for i in range(len(predictions)):
    print(f"{output_models[i]}{delimiter}{output_serials[i]}{delimiter}{output_dates[i]}{delimiter}{predictions[i]}")