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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline

# Main
# Task 1: Regression on the Computer Hardware Dataset
hardwareData = pd.read_csv('machine.csv')  # Read in data form csv file

# Complete attribute list & statistical attribute list
all_attributes = ['vendor', 'model', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP']
stat_attributes = all_attributes[2:]
 # Set up a list to store column labels (numbered columns)
hardwareData.columns = ["column" + attribute for attribute in all_attributes] + ["ERP"]

correlation_dic = {}
# Get a list of each attributes correlation coefficent to ERP
for i in stat_attributes:
    correlation = round(stats.correlation(hardwareData['ERP'], hardwareData[i]), ndigits=6)
    correlation_dic[i] = correlation
    print(i + "\nCorrelation Coefficient to ERP: " + str(correlation))

# Get sorted list of highest correlations to lowest as a key value pair, sorted by value
sorted_correlation = dict(sorted(correlation_dic.items(), key=lambda item: item[1], reverse=True)) 
top_attributes = list(sorted_correlation.keys())[:4]  # Attribute names of top 4 correlations

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
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print(comparison)