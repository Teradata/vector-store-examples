# Import the required libraries
import sys
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Read input from STDIN
delimiter = '\t'
inputData = []

for line in sys.stdin.read().splitlines():
    line = line.split(delimiter)
    inputData.append(line)

# Convert input data to a pandas DataFrame
columns = ['userID', 'productID', 'rating']
pandas_df = pd.DataFrame(inputData, columns=columns).copy()
del inputData

# Convert the columns to numeric
for c in columns:
    pandas_df[c] = pd.to_numeric(pandas_df[c])

# Create user-product matrix
user_product_matrix = pandas_df.pivot_table(index='userID', columns='productID', values='rating').fillna(0)

# Split into train and test sets
train_data, test_data = train_test_split(user_product_matrix, test_size=0.2, random_state=42)

# Apply TruncatedSVD
svd = TruncatedSVD(n_components=180, random_state=42)
normalizer = Normalizer()
model = make_pipeline(svd, normalizer)

# Fit the model
model.fit(train_data)

# Use only SVD for reconstruction
svd_only = TruncatedSVD(n_components=180, random_state=42)
svd_only.fit(train_data)
test_projected = svd_only.transform(test_data)
test_reconstructed = svd_only.inverse_transform(test_projected)

# Compute metrics
mse = mean_squared_error(test_data.values, test_reconstructed)
explained_variance = svd_only.explained_variance_ratio_.sum()

# Output metrics in tab-separated format
print(f"{explained_variance}\t{mse}")
