"""
Kyle Tranfaglia
COSC311 - Project02
Last updated 05/07/24
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
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.preprocessing import (StandardScaler, PolynomialFeatures,  MinMaxScaler, RobustScaler,
                                   MaxAbsScaler, PowerTransformer, QuantileTransformer, Normalizer)

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import mode


# Sliding window function to segment data
def sliding_window(df, win_len):
    segments = []
    num_samples = len(df)
    # Iterate over the data with a step size of window length
    for i in range(0, num_samples - win_len + 1, win_len):
        segment = df.iloc[i:i + win_len]  # Get segment of data including 'window_length' consecutive samples
        segments.append(segment)  # Add the segment to the list of segments
    return segments


# Determine the best window length for segmentation
def best_window_length(act_data, all_labels):
    window_lengths = range(64, 1025, 64)  # Define window length range to run test
    best_accuracy = 0
    best_window_length = 0
    for win_len in window_lengths:
        samples = []
        labels = []
        for i, df in enumerate(act_data):
            segments = sliding_window(df, win_len)
            for segment in segments:
                features = segment.values.flatten()
                samples.append(features)
                labels.append(all_labels[i])

        # Normalize the features
        scaler = StandardScaler()
        samples_normalized = scaler.fit_transform(samples)

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(samples_normalized, labels, test_size=0.2, random_state=7)

        # Train a classifier
        svc = SVC(kernel='linear', C=6, gamma='scale', random_state=7)
        svc.fit(x_train, y_train)

        # Evaluate the classifier
        prediction = svc.predict(x_test)
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


# Evaluate classifiers using self test
def evaluate_self_test(classifier, features, labels):
    classifier.fit(features, labels)  # Fit the classifier on the entire dataset
    predictions = classifier.predict(features)  # Make predictions
    accuracy = accuracy_score(labels, predictions)  # Calculate accuracy

    return accuracy


# Evaluate classifiers using independent test
def evaluate_independent_test(classifier, features_train, labels_train, features_test, labels_test):
    classifier.fit(features_train, labels_train)  # Fit the classifier on the training data
    predictions = classifier.predict(features_test)  # Make predictions on the test data
    accuracy = accuracy_score(labels_test, predictions)  # Calculate accuracy

    return accuracy


# Evaluate classifiers using cross-validation test
def evaluate_cross_validation(classifier, features, labels):
    scores = cross_val_score(classifier, features, labels, cv=10)  # Use cross_val_score to perform cross-validation
    average_accuracy = np.mean(scores)  # Calculate the average accuracy

    return average_accuracy


# Main

# Task 1: Data Segmentation

# Load CSV files into DataFrames
files = ["COUGH.csv", "DRINK.csv", "EAT.csv", "READ.csv", "SIT.csv", "WALK.csv"]
activity_data = [pd.read_csv(file, skiprows=1) for file in files]

all_labels = ["COUGH", "DRINK", "EAT", "READ", "SIT", "WALK"]  # Define the list of all activity labels

# Compute and display the best window length (one that yields highest model accuracy)
window_length = best_window_length(activity_data, all_labels)
print("Best Window Length Found:", window_length)

# window_length = 512  # Define window length

# Segment data into a list (of lists)
segmented_data = [sliding_window(dataset, window_length) for dataset in activity_data]

# Task 2: Feature Extraction

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

# Combine features and labels for each segment to generate samples
samples = np.array(features_per_segment)
labels = np.array(labels_per_segment)

# Normalize the features
scaler = MinMaxScaler()
samples_normalized = scaler.fit_transform(samples)

# Define classifiers
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=1),
    "Random Forest": RandomForestClassifier(n_estimators=315, criterion='gini', max_depth=14, min_samples_split=3,
                                            min_samples_leaf=3, max_features='sqrt', random_state=7),
    "SVC": SVC(kernel='linear', C=6, probability=True, gamma='scale', random_state=7),
    "MLP": MLPClassifier(hidden_layer_sizes=100, activation='tanh', solver='adam', alpha=1e-5, batch_size=36, tol=1e-6,
                         learning_rate_init=0.01, learning_rate='constant', max_iter=10000, random_state=7),
    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=7),
    "Polynomial LR": make_pipeline(PolynomialFeatures(2), LogisticRegression(solver='liblinear', random_state=7))
}

# Add VotingClassifier separately in order to use all classifiers in dict for voting
voting_classifier = VotingClassifier(estimators=list(classifiers.items()), voting='soft')
classifiers["Voting"] = voting_classifier

for classifier_name, classifier_obj in classifiers.items():
    # Experiment 1: With feature normalization
    scores_with_normalization = cross_val_score(classifier_obj, samples_normalized, labels, cv=6)

    # Experiment 2: Without feature normalization
    scores_without_normalization = cross_val_score(classifier_obj, samples, labels, cv=6)

    # Compare and display the average performance metrics
    print("Test Classifier:", classifier_name)
    print("Average accuracy with feature normalization:", scores_with_normalization.mean())
    print("Average accuracy without feature normalization:", scores_without_normalization.mean())

# Task 4: Model Training and Testing

# Split the data into training and testing subsets for independent testing
x_train, x_test, y_train, y_test = train_test_split(samples_normalized, labels, test_size=0.2, random_state=7)

# Evaluate classifiers using different evaluation methods
for classifier_name, classifier_obj in classifiers.items():
    self_test_results = evaluate_self_test(classifier_obj, samples_normalized, labels)
    independent_test_results = evaluate_independent_test(classifier_obj, x_train, y_train, x_test, y_test)
    cross_validation_results = evaluate_cross_validation(classifier_obj, samples_normalized, labels)

    # Compare and display the average performance metrics
    print("Test Classifier:", classifier_name)
    print("Average accuracy with self-test:", self_test_results)
    print("Average accuracy with independent-test:",  independent_test_results)
    print("Average accuracy with cross-validation:", cross_validation_results)
