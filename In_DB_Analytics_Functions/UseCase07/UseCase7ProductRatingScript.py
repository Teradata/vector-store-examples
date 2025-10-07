# Import the required libraries
import sys
import pandas as pd
import numpy as np
#import os
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Read input from STDIN
delimiter = '\t'
train_data = []
test_data = []
#print('Test 1')

for line in sys.stdin.read().splitlines():
    line = line.split(delimiter)
    if len(line) < 4:
        continue
    try:
        train_ind = int(line[0])
        userID = int(line[1])
        productID = int(line[2])
        rating = int(line[3])
        if train_ind == 0:
            test_data.append([userID, productID, rating])
        else:
            train_data.append([userID, productID, rating])
    except ValueError:
        continue


# Turn train and test sets into numpy arrays
train_data = np.array(train_data)
test_data = np.array(test_data)
#print(train_data.shape)
#print(test_data.shape)
#print(f"{train_data.shape}\t{test_data.shape}")

# Extract the row, column, and value arrays
userIDs_train = train_data[:, 0]
productIDs_train = train_data[:, 1]
ratings_train = train_data[:, 2]
#print(userIDs_train, productIDs_train, ratings_train)

userIDs_test = test_data[:, 0]
productIDs_test = test_data[:, 1]
ratings_test = test_data[:, 2]
#print(userIDs_test, productIDs_test, ratings_test)

# Create mappings to contiguous indices
unique_userIDs = np.unique(np.concatenate((userIDs_train, userIDs_test)))
unique_productIDs = np.unique(np.concatenate((productIDs_train, productIDs_test)))

userID_map = {id_: idx for idx, id_ in enumerate(unique_userIDs)}
productID_map = {id_: idx for idx, id_ in enumerate(unique_productIDs)}

# Apply mappings
mapped_userIDs_train = np.array([userID_map[id_] for id_ in userIDs_train])
mapped_productIDs_train = np.array([productID_map[id_] for id_ in productIDs_train])

mapped_userIDs_test = np.array([userID_map[id_] for id_ in userIDs_test])
mapped_productIDs_test = np.array([productID_map[id_] for id_ in productIDs_test])

# Define matrix shape
num_users = len(unique_userIDs)
num_products = len(unique_productIDs)
matrix_shape = (num_users, num_products)

# Create sparse matrices
matrix_train = coo_matrix((ratings_train, (mapped_userIDs_train, mapped_productIDs_train)), shape=matrix_shape)
matrix_test = coo_matrix((ratings_test, (mapped_userIDs_test, mapped_productIDs_test)), shape=matrix_shape)

# Apply TruncatedSVD
svd = TruncatedSVD(n_components=180, random_state=42)
normalizer = Normalizer()
model = make_pipeline(svd, normalizer)

# Fit the model
model.fit(matrix_train)

# Use only SVD for reconstruction
svd_only = TruncatedSVD(n_components=180, random_state=42)
svd_only.fit(matrix_train)
test_projected = svd_only.transform(matrix_test)
test_reconstructed = svd_only.inverse_transform(test_projected)

# Compute metrics
mse = mean_squared_error(matrix_test.toarray().flatten(), test_reconstructed.flatten())
explained_variance = svd_only.explained_variance_ratio_.sum()

# Output metrics in tab-separated format
print(f"{explained_variance}\t{mse}")