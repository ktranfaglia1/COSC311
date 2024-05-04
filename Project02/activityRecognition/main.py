"""
Kyle Tranfaglia
COSC311 - Project02
Last updated 05/3/24
Task 1: Data Segmentation - segmentation of data with sliding window.
Task 2: Feature Extraction - Each segment is used to extract multiple features to represent an activity.
Task 3: Dataset Generation - Combination of all features and corresponding activity labels to generate sample. Features
normalized before model train and test
Task 4: Model Training and testing - Experiment to compare classifiers and find the best classifier for the dataset.
The best classifier is denoted by best overall performance in modeling data (accurate labeling)
Task 5: Experience and Potential Improvements - Reflection of project
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import mode


# KNN (K-Nearest Neighbors) Algorithm to fit the classifier to the training data
# and target labels and return the predictions for the test data
def knnClassifier(attributeTrain, attributeTest, targetTrain):
    # Instantiate a KNeighborsClassifier object with 1 neighbors, optimal for data set
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(attributeTrain, targetTrain)  # Fit the classifier to the training data and target labels

    return knn.predict(attributeTest)  # Return the predictions for the test data


# MLP (Multi-Layer Perceptron) Algorithm to fit the classifier to the training data
# and target labels and return the predictions for the test data
def mlpClassifier(attributeTrain, attributeTest, targetTrain):
    # Instantiate an MLPClassifier object with optimized parameters
    mlp = MLPClassifier(hidden_layer_sizes=100, activation='tanh', solver='adam', alpha=1e-5, batch_size=36, tol=1e-6,
                        learning_rate_init=0.01, learning_rate='constant', max_iter=10000, random_state=7)
    mlp.fit(attributeTrain, targetTrain)  # Fit the classifier to the training data and target labels

    return mlp.predict(attributeTest)  # Return the predictions for the test data


# RF (Random Forest) Algorithm to fit the classifier to the training data and target
# labels and return the predictions for the test data
def rfClassifier(attributeTrain, attributeTest, targetTrain):
    # Instantiate an RFClassifier object with optimized parameters
    rf = RandomForestClassifier(n_estimators=315, criterion='gini', max_depth=14, min_samples_split=3,
                                min_samples_leaf=3, max_features='sqrt', random_state=7)
    rf.fit(attributeTrain, targetTrain)  # Fit the classifier to the training data and target labels

    return rf.predict(attributeTest)  # Return the predictions for the test data


# SVC (Support Vector Classifier) Algorithm to fit the classifier to the training
# data and target labels and return the predictions for the test data
def svcClassifier(attributeTrain, attributeTest, targetTrain):
    # Instantiate an SVCClassifier object with optimized parameters
    svc = SVC(kernel='rbf', C=13, gamma='scale', random_state=7)
    svc.fit(attributeTrain, targetTrain)  # Fit the classifier to the training data and target labels

    return svc.predict(attributeTest)  # Return the predictions for the test data


# Logistic Regression Algorithm to fit the classifier to the training data and target
# labels and return the predictions for the test data
def logRegClassifier(attributeTrain, attributeTest, targetTrain):
    # Instantiate an LogisticRegressionClassifier object with optimized parameters
    logReg = LogisticRegression(solver='liblinear', random_state=7)
    logReg.fit(attributeTrain, targetTrain)  # Fit the classifier to the training data and target labels

    return logReg.predict(attributeTest)  # Return the predictions for the test data


# Pipelined Logistic Regression and polynomial feature transformation Algorithm to fit the
# classifier to the training data and target labels and return predictions
def pipelineClassifier(attributeTrain, attributeTest, targetTrain):
    # Note: Increasing polynomial features count improves accuracy but significantly decreases performance
    # Instantiate a PiplinedClassifier object to combine polynomial feature transformation with
    # logistic regression with optimized parameters
    pipeline = make_pipeline(PolynomialFeatures(2), LogisticRegression(solver='liblinear', random_state=7))
    pipeline.fit(attributeTrain, targetTrain)  # Fit the classifier to the training data and target labels

    return pipeline.predict(attributeTest)  # Return the predictions for the test data


# Voting Algorithm to fit the classifier to the training data and target labels and return the predictions
# Uses all the other classifiers and uses a weighted average of predicted probabilities for create a prediction
def votingClassifier(attributeTrain, attributeTest, targetTrain):
    # All Classifiers for voting
    knn = KNeighborsClassifier(n_neighbors=1)
    mlp = MLPClassifier(hidden_layer_sizes=100, activation='tanh', solver='adam', alpha=1e-5, batch_size=36, tol=1e-6,
                        learning_rate_init=0.01, learning_rate='constant', max_iter=10000, random_state=7)
    rf = RandomForestClassifier(n_estimators=315, criterion='gini', max_depth=14, min_samples_split=3,
                                min_samples_leaf=3, max_features='sqrt', random_state=7)
    svc = SVC(kernel='rbf', C=13, gamma='scale', probability=True, random_state=7)
    pipeline = make_pipeline(PolynomialFeatures(2), LogisticRegression(solver='liblinear', random_state=7))

    # Create a voting ensemble of the classifiers with soft voting. Weighted average of predicted probabilities
    votingSystem = VotingClassifier(
        estimators=[('mlp', mlp), ('knn', knn), ('svc', svc), ('rf', rf), ('pipeline', pipeline)], voting='soft')
    votingSystem.fit(attributeTrain, targetTrain)  # Fit the classifier to the training data and target labels

    return votingSystem.predict(attributeTest)  # Return the predictions for the test data


# Determine the best window length for segmentation
def bestWindowLength(act_data):
    all_labels = ["COUGH", "DRINK", "EAT", "READ", "SIT", "WALK"]
    window_lengths = range(128, 1025, 128)  # Define window length range to run test
    best_accuracy = 0
    best_window_length = 0
    for win_len in window_lengths:
        samples = []
        labels = []
        for i, df in enumerate(act_data):
            segments = sliding_window(df, win_len)
            for segment in segments:
                features = segment[['X', 'Y', 'Z']].values.flatten()
                samples.append(features)
                labels.append(all_labels[i])

        # Normalize the features
        scaler = StandardScaler()
        samples_normalized = scaler.fit_transform(samples)

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(samples_normalized, labels, test_size=0.2, random_state=7)

        # Train a classifier
        clf = RandomForestClassifier(random_state=7)
        clf.fit(x_train, y_train)

        # Evaluate the classifier
        prediction = clf.predict(x_test)
        accuracy = accuracy_score(y_test, prediction)

        # Update the best window length if the current one has higher accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_window_length = win_len

    return best_window_length


# Extract the features
def extract_features(segment):
    features = []

    # Statistical features
    features.extend(segment.mean(axis=0))  # Mean along each axis
    features.extend(segment.std(axis=0))  # Standard deviation along each axis
    features.extend(segment.max(axis=0))  # Maximum value along each axis
    features.extend(segment.min(axis=0))  # Minimum value along each axis
    features.extend(np.sqrt(np.mean(segment ** 2, axis=0)))  # Root Mean Square (RMS) along each axis

    return features


# Sliding window function to segment data
def sliding_window(df, win_len):
    segments = []
    num_samples = len(df)
    # Iterate over the data with a step size of window length
    for i in range(0, num_samples - win_len + 1, win_len):
        segment = df.iloc[i:i + win_len]  # Get segment of data including 'window_length' consecutive samples
        segments.append(segment)  # Add the segment to the list of segments
    return segments


# Main

# Task 1: Data Segmentation
# Load CSV files into DataFrames
files = ["COUGH.csv", "DRINK.csv", "EAT.csv", "READ.csv", "SIT.csv", "WALK.csv"]
activity_data = [pd.read_csv(file) for file in files]

# Set up a list to store column labels, then apply labels
all_attributes = ['X', 'Y', 'Z']
activity_data.columns = [attribute for attribute in all_attributes]

# window_length = bestWindowLength(activity_data)
# segmented_data = [sliding_window(dataset, window_length) for dataset in activity_data]

window_length = 512  # Define window length

# Segment data into a list (of lists)
segmented_data = [sliding_window(dataset, window_length) for dataset in activity_data]

# Task 2: Feature Extraction
all_labels = ["COUGH", "DRINK", "EAT", "READ", "SIT", "WALK"]  # Define the list of all activity labels
# Store features and labels for each segment
features_per_segment = []
labels_per_segment = []

# Iterate over each dataset (list of segments) in segmented_data
for i, dataset_segments in enumerate(segmented_data):
    # Iterate over each segment in the current dataset
    for segment in dataset_segments:
        features = extract_features(segment)  # Extract features for the current segment
        features_per_segment.append(features)  # Append the extracted features to the features_per_segment list
        labels_per_segment.append(all_labels[i])  # Append the corresponding label to the labels_per_segment list

# Task 3: Dataset Generate
