"""
Kyle Tranfaglia
COSC311 - Project02
Last updated 05/3/24
Task 1: Data segmentation - segmentation of data with sliding window.
Task 2: Feature extraction - Each segment is used to extract multiple features to represent an activity.
Task 3: Dataset generation - Combination of all features and corresponding activity labels to generate sample. Features
normalized before model train and test
Task 4: Model training and testing - Experiment to compare classifiers and find the best classifier for the dataset.
The best classifier is denoted by best overall performance in modeling data (accurate labeling)
Task 5: Experience and potential improvements - Reflection of project
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import mode


# Main

# Task 1: Data segmentation

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
