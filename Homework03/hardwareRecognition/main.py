'''
Kyle Tranfaglia
COSC311 - Homework03
Last updated 03/26/24
Task 1: Regression on the Computer Hardware Dataset
This program reads in a computer hardware dataset, then measures the correlation between each attribute and the "ERP" to extract the four
best attributes. With these attributes and the "ERP," the data is randomly split into 60% training data and 40% testing data. Finally, with
the training data, a multiple linear regression model is built and evaluated using the testing data. The results show the MAE, MSE, and RMSE.
Task 2: Clustering on Hand-Written Digits
The program reads in the UCI ML hand-written digits dataset, then conducts a PCA analysis on the dataset and finds m, which is how many 
principal components are needed to keep at least 85% variance. Next, the dataset is transformed from 64 dimensions to m dimensions. With the 
dimension-reduced dataset k-means clustering is conducted and the center of each cluster is displayed. Finally, the learned label is matched 
to the true label, and then clustering accuracy is calculated and the corresponding confusion matrix is displayed.
'''
import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
'''
# Main
# Task 1: Regression on the Computer Hardware Dataset
hardwareData = pd.read_csv('machine.data')  # Read in data form csv file

# Complete attribute list & statistical attribute list
all_attributes = ['vendor', 'model', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP']
stat_attributes = all_attributes[2:]
 # Set up a list to store column labels (numbered columns)
hardwareData.columns = [attribute for attribute in all_attributes] + ["ERP"]

correlation_dic = {}
print("\nAttributes correlation coefficent to ERP")
# Get a list and print each attributes correlation coefficent to ERP
for i in stat_attributes:
    correlation = round(stats.correlation(hardwareData['ERP'], hardwareData[i]), ndigits=6)
    correlation_dic[i] = correlation
    print(i + " Correlation Coefficient to ERP: " + str(correlation))

# Get sorted list of highest correlations to lowest as a key value pair, sorted by value
sorted_correlation = dict(sorted(correlation_dic.items(), key=lambda item: item[1], reverse=True)) 
top_attributes = list(sorted_correlation.keys())[:4]  # Attribute names of top 4 correlations
print("Top Attributes:", top_attributes)

# Extract features and target data
attribute_data = hardwareData[top_attributes]
target_data = hardwareData['ERP']

# Split the data into training and testing sets, 60% training, 40% testing
attribute_train, attribute_test, target_train, target_test = train_test_split(attribute_data, target_data, test_size=0.4, random_state=7)

# Fit multiple linear regression model
lrModel = LinearRegression()
lrModel.fit(attribute_train, target_train)

# Predict target variable using testing data & generate a comparative table for target predictions and actual 
predictions = lrModel.predict(attribute_test)
comparison = pd.DataFrame({"Prediction":predictions, "Actual":target_test})

# Calculate evaluation metrics: MAE, MSE, RMSE
mae = mean_absolute_error(target_test, predictions)
mse = mean_squared_error(target_test, predictions)
rmse = np.sqrt(mse)

# Print the evaluation metrics
print("\nEvaluation Matrics")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print(comparison)
'''
# Task 2: Clustering on Hand-Written Digits

# Load the hand-written digits dataset
digits = load_digits()
digits_data = digits.data

# Standardize the matrix
digits_mean = np.mean(digits_data, axis=0)
digits_std = np.std(digits_data, axis=0)
digits_std[digits_std == 0] = 1e-10  # Change stds of 0 to a small value to avoid division by zero
digits_normalized = (digits_data - digits_mean) / digits_std

# Perform Principal Component Analysis (PCA) with normalized data
pca_normalized = PCA(n_components=0.85, svd_solver='full')
digits_data_new = pca_normalized.fit_transform(digits_normalized)

# Get the covariance matrix with normalized data
covariance_matrix = pca_normalized.get_covariance()

# Dsiplay covariance matrix with normalized data
print("PCA Analysis with normalized data")
print("Covariance Matrix:")
print(covariance_matrix)

# Number of principal components required to keep at least 85% variance with normalized data
print("Percentage of variance explained by each component to the total variance:\n", pca_normalized.explained_variance_ratio_)
print(f"Total explained variance ratio: {np.sum(pca_normalized.explained_variance_ratio_):.2f}")
print(f"Number of principal components to keep at least 85% variance: {pca_normalized.n_components_}")

# Perform Principal Component Analysis (PCA) without normalized data
pca_unnormalized  = PCA(n_components=0.85, svd_solver='full')
digits_data_new = pca_unnormalized.fit_transform(digits_data)

# Get the covariance matrix without normalized data
covariance_matrix = pca_unnormalized.get_covariance()

# Dsiplay covariance matrix without normalized data
print("PCA Analysis without normalized data")
print("Covariance Matrix:")
print(covariance_matrix)

# Number of principal components required to keep at least 85% variance without normalized data
print("Percentage of variance explained by each component to the total variance:\n", pca_unnormalized.explained_variance_ratio_)
print(f"Total explained variance ratio: {np.sum(pca_unnormalized.explained_variance_ratio_):.2f}")
print(f"Number of principal components to keep at least 85% variance: {pca_unnormalized.n_components_}")

# Perform PCA
pca_final = PCA(n_components=17)
digits_data_transformed = pca_final.fit_transform(digits_normalized)

# Check the shape of the transformed dataset
print("Shape of the transformed dataset:", digits_data_transformed.shape)

# Perform k-means clustering
kmeans = KMeans(n_clusters=10, random_state=7)
kmeans.fit(digits_data_transformed)

# Output the center of each cluster
print("Center of each cluster (each cluster represents a digit):")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i}: {center:.6f}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits_data_transformed, digits.target, test_size=0.3, random_state=7)

# Initialize the KNeighborsClassifier object
knn_pca = KNeighborsClassifier(n_neighbors=4)  # Use number of samples in the training set as neighbors

# Fit the model to the training data using the actual target labels
knn_pca.fit(X_train, y_train)

# Output the center of each cluster
print("Center of each cluster (each cluster represents a digit):", knn_pca._fit_X)
print(f"Train score after PCA: {knn_pca.score(X_train, y_train):.6f}")
print(f"Test score after PCA: {knn_pca.score(X_test, y_test):.6f}")

# Determine the mapping between cluster labels and true labels
cluster_labels = kmeans.labels_
cluster_mapping = {}
for cluster in np.unique(cluster_labels):
    mask = (cluster_labels == cluster)
    true_labels = digits.target[mask]
    mode_label = mode(true_labels, keepdims=True)[0][0]
    cluster_mapping[cluster] = mode_label

# Calculate clustering accuracy
predicted_labels = [cluster_mapping[cluster] for cluster in cluster_labels]
accuracy = np.mean(predicted_labels == digits.target)

print(f"Clustering Accuracy: {accuracy:.6f}")

# Generate confusion matrix
confusionMatrix = confusion_matrix(digits.target, predicted_labels)

# Visualize the confusion matrix using matplotlib and the sklearn toolset
matrixDisplay = ConfusionMatrixDisplay(confusion_matrix = confusionMatrix)
fig, ax = plt.subplots(figsize=(10, 8))  # Create layout and structure figure
matrixDisplay.plot(ax = ax, cmap = 'Blues')  # Create Plot
# Plot labels
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Clustering Confusion Matrix')
plt.show()