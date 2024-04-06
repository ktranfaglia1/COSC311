'''
Kyle Tranfaglia
COSC311 - Project01
Last updated 04/06/24
This program uses the "Wireless Indoor Localization Data Set" to perform a Self-test for KNN and DT algorithms,
Independent-test for KNN and DT algorithms, and Classification model finalization
'''
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Print the confusion matrix, classification report, accuracy score, heatmap, and matrix display
def printResults(targetTest, prediction):
    confusionMatrix = confusion_matrix(targetTest, prediction)  # Create confusion matrix

    # Display the confusion matrix, a classification report, and the overall accuracy score of the prediction
    print("\nConfusion Matrix:\n", confusionMatrix)
    print("\nClassification Report:\n", classification_report(targetTest, prediction))
    print("Accuracy Score:", accuracy_score(targetTest, prediction))

    # Display a heatmap using matplotlib and the sklearn toolset to display a confusion matrix
    matrixDisplay = ConfusionMatrixDisplay(confusion_matrix = confusionMatrix)
    fig, ax = plt.subplots(figsize=(10, 8))  # Create layout and structure figure
    matrixDisplay.plot(ax = ax, cmap = 'Blues')  # Create Plot
    # Plot labels
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Voting Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save plot as png
    # plt.show()  # Display plot

    # Display Heatmap using Seaborn
    # sb.heatmap(confusionMatrix, annot = False, fmt = 'd', cmap = 'Blues', cbar = False)  # Create heatmap with Seaborn
    # plt.savefig('heatmap.png')  # Save plot as png
    # plt.show()  # Display plot

# Main
# Read in data
wifi_data = pd.read_csv("wifi_localization.txt", sep="\t") 
print(wifi_data.info())

# Create K Nearest Neighbors and Decision Tree classifiers
knn_classifier = KNeighborsClassifier(n_neighbors=5, algorithm="brute", weights="uniform", random_state=7)
dt_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=9, min_samples_split=2, random_state=7)

# Task 1 - Self-test
# Extract features and target labels
self_test_data = wifi_data.iloc[:, :-1]
self_test_target = wifi_data.iloc[:, -1]

# Scale the data
scaler = StandardScaler()
self_test_data_scaled = scaler.fit_transform(self_test_data)

# Fit the classifiers to the self-test data
knn_classifier.fit(self_test_data_scaled, self_test_target)
dt_classifier.fit(self_test_data_scaled, self_test_target)

# Get predictions
knn_self_test_predict = knn_classifier.predict(self_test_data_scaled)
dt_self_test_predict = dt_classifier.predict(self_test_data_scaled)

# Print results for self-test
printResults(self_test_target, knn_self_test_predict)  # knn results
printResults(self_test_target, dt_self_test_predict)  # dt results

# Task 2 - Independent-test
# Split the data into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(wifi_data.iloc[:, :-1], wifi_data.iloc[:, -1], test_size=0.3, random_state=7)

# Scale the training and testing data
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Fit the classifiers to the independent-test data
knn_classifier.fit(train_data_scaled, train_target)
dt_classifier.fit(train_data_scaled, train_target)

# Get predictions
knn_independent_test_predict = knn_classifier.predict(test_data_scaled)
dt_independent_test_predict = dt_classifier.predict(test_data_scaled)

# Print results for independent-test
printResults(test_target, knn_independent_test_predict)  # knn results
printResults(test_target, dt_independent_test_predict)  # dt results