'''
Kyle Tranfaglia
COSC311 - Project01
Last updated 04/08/24
This program uses the "Wireless Indoor Localization Data Set" to perform a Self-test for KNN and DT algorithms,
Independent-test for KNN and DT algorithms, and Classification model finalization
'''
import pandas as pd
import matplotlib.pyplot as plt
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
    matrixDisplay.plot(ax = ax, cmap = 'Reds')  # Create Plot
    # Plot labels
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    # plt.savefig('confusion_matrix.png')  # Save plot as png
    plt.show()  # Display plot

    # Display Heatmap using Seaborn
    # sb.heatmap(confusionMatrix, square = True, annot = True, fmt = 'd', cmap = 'Blues', cbar = False)  # Create heatmap with Seaborn
    # plt.savefig('heatmap.png')  # Save plot as png
    # plt.show()  # Display plot

# Main
# Read in data and label
wifi_data = pd.read_csv("wifi_localization.txt", sep="\t")
wifi_data.columns = ["column" + str(i + 1) for i in range(0, len(wifi_data.columns) - 1)] + ["Target"]  # Set up a list to store column labels (numbered columns)
print(wifi_data.info())

# Pre-test data analysis and set up

# Extract features and target labels
self_test_data = wifi_data.iloc[:, :-1].values
self_test_target = wifi_data.iloc[:, -1].values

# Scale the data
scaler = StandardScaler()
self_test_data_scaled = scaler.fit_transform(self_test_data)

wifi_data.plot()
# Plot labels
plt.xlabel('Data Column')
plt.ylabel('Data Value')
plt.title('Wifi Localization Data Distribution')
plt.show()  # Display plot

# Iterate over each feature in the self_test_data to show data distribution
for i in range(0, 7):
    plt.scatter(self_test_data[:, i], self_test_data[:, i+1 if i < 6 else 0], s=40)  # Create a scatter plot between the i and the i+1 feature
plt.show()  # Plot scatter

# Declare variables for accuracy calculation and storage
neighbors = np.arange(1, 15)
accuracy = []

# Test different neighbor values to find the optimal number for the dataset
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)  # Create K Nearest Neighbors classifier with k neighbors
    knn.fit(self_test_data_scaled, self_test_target)  # Fit the classifiers to the self-test data
    accuracy.append(knn.score(self_test_data_scaled, self_test_target))  # Append accuracy results to accuracy array

# Set up line graph
plt.figure(figsize=(10, 8))
plt.plot(neighbors, accuracy)
plt.title('KNN Self-test Accuracy by Neighbor Count')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

# Task 1 - Self-test

# Create K Nearest Neighbors and Decision Tree classifiers
# knn_classifier = KNeighborsClassifier(n_neighbors=5, algorithm="brute", weights="uniform")
knn_classifier = KNeighborsClassifier(n_neighbors=4, algorithm="brute", weights='distance')
# dt_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=9, min_samples_split=2)
dt_classifier = DecisionTreeClassifier(criterion='gini', class_weight='balanced', max_features=3, max_depth=16, min_samples_split=2)

# Fit the classifiers to the self-test data
knn_classifier.fit(self_test_data_scaled, self_test_target)
dt_classifier.fit(self_test_data_scaled, self_test_target)

# Get predictions
knn_self_test_predict = knn_classifier.predict(self_test_data_scaled)
dt_self_test_predict = dt_classifier.predict(self_test_data_scaled)

# Print results for self-test
print("Self-test KNN Results:")
printResults(self_test_target, knn_self_test_predict)  # knn results
print("Self-test DT Results:")
printResults(self_test_target, dt_self_test_predict)  # dt results

'''
Varying numbers of neighbors, the algorithm type, and the weights for KNN was tested. Also, the max_features, min_samples_split, 
max_depth, criterion, and class_weight for the DT were tested in order to determine the best combination of parameters to 
produce the highest self test accuracy. From the data analysis, it was determined that k=4 was the optimal amount of neighbors
for the KNN algorithm with "brute" as the algorithm and "distance" as the weights. Relative to the neighbors, these parameters 
had minor impacts on the accuracy, although they were able to bring the accuracy to 100% for self-test. As for DT, max_depth 
seemed to have the greatest impact on accuracy such that increasing the max_depth value increased accuracy. It was found that 
a value of 16 for max_depth yielded the greatest accuracy. Additionally, it was found that the best value for the min_sample_splits
was 2, max_features was 3, class_weight was "balanced," and criterion was "gini." It is important to note that the parameters beyond 
max_depth had minor impact on accuracy, however, the combined parameters brings the DT classifier to 100% accuracy for self-test.
'''

# Task 2 - Independent-test

# Split the data into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(wifi_data.iloc[:, :-1], wifi_data.iloc[:, -1], test_size=0.3, random_state=7)

# Scale the training and testing data
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Fit the classifiers to the independent-test data using same classifiers from task 1
knn_classifier.fit(train_data_scaled, train_target)
dt_classifier.fit(train_data_scaled, train_target)

# Get predictions
knn_independent_test_predict = knn_classifier.predict(test_data_scaled)
dt_independent_test_predict = dt_classifier.predict(test_data_scaled)

# Print results for independent-test
print("Independent test KNN Results:")
printResults(test_target, knn_independent_test_predict)  # knn results
print("Independent test DT Results:")
printResults(test_target, dt_independent_test_predict)  # dt results

'''
The same classifier models, as in the KNN and DT parameters were used for the self test and the independent test. It was found 
that this combination of parameters was optimal for both tests such that the accuracy was the greatest for both the self-test
and independent test when using the current parameters. The same parameters that were tested in task 1 were tested in task 2
as well, however, no changes to the model were able to surpass the accuracy scores of the current model. Therefore, the model
was left unchanged moving from the self-test to the independent test as it appears the model is optimized for both tests.

For KNN, the model had a final accuracy score of 98% with a 0.98 for precision, recall, and f1-score, which is relatively high for 
a machine learning algorithm. The Decision tree model was not far behind with a final accuracy score of 96.67% with a 0.97 for 
precision, recall, and f1-score, which is still relatively high, but not not as good as the KNN model, despite model optimization. 
Overall, both models performed well in the independent test, with KNN being slightly more accurate, and neither algorithm could
be further optimized in transition from the self-test to the independent test in regards to the model parameters. 
'''

# Task 3 - Classification model finalization

# Final optimization 
# knn_classifier = KNeighborsClassifier(n_neighbors=4, algorithm="brute", weights="distance")  # Create K Nearest Neighbors classifier

# # Split the data into training and testing sets
# train_data, test_data, train_target, test_target = train_test_split(wifi_data.iloc[:, :-1], wifi_data.iloc[:, -1], test_size=0.3, random_state=7)

# # Scale the training and testing data
# train_data_scaled = scaler.fit_transform(train_data)
# test_data_scaled = scaler.transform(test_data)

# # Fit the classifiers to the independent-test data
# knn_classifier.fit(train_data_scaled, train_target)
# dt_classifier.fit(train_data_scaled, train_target)

# # Get predictions and print results
# knn_predict = knn_classifier.predict(test_data_scaled)
# printResults(test_target, knn_predict)

'''
Final optimization: KNN is the best classifier for the dataset. Upon previous testing and further parameter manipulation, the current KNN classifier
with the given parameters is the most optimized version: KNeighborsClassifier(n_neighbors=4, algorithm="brute", weights="distance")
This optimized classifier reaches 98% accuracy with random state of 7 for train_test_split, which is the greatest accuracy achieved.
KNN was tested with all the algorithm options and weight options, and the accuracy of 1-14 neighbors was tested to ensure the current 
version was the most optimized. Given the 98% accuracy, relative to the 96.67% accuracy for the Decision Tree classifier post optimization,
the KNN classifier will be used to conduct further testing with various train and test data splits.
'''

# Lists to store test sizes and corresponding accuracies
test_sizes, accuracies = [0.1, 0.2, 0.3, 0.4, 0.5], []

# KNN classifier for all test sizes
for i in test_sizes:
    # Split the data into training and testing sets
    train_data, test_data, train_target, test_target = train_test_split(wifi_data.iloc[:, :-1], wifi_data.iloc[:, -1], test_size=i, random_state=7)

    # Scale the training and testing data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    knn_classifier.fit(train_data_scaled, train_target)  # Fit the classifier to the training data
    knn_predict = knn_classifier.predict(test_data_scaled)  # Get predictions
    accuracies.append(accuracy_score(test_target, knn_predict))  # Calculate accuracy
    print("Independent test " + str(i) + " KNN Results:")
    printResults(test_target, knn_predict)  # knn results

# Plot the accuracies
plt.figure(figsize=(10, 8))
plt.bar(test_sizes, accuracies, width=0.05)
# Add text labels for accuracy values at the top of each bar for closer analysis
for i, accuracy in enumerate(accuracies):
    plt.text(test_sizes[i], accuracy, f'{accuracy:.3f}', ha='center', va='bottom')
plt.xlabel('Test Size')
plt.ylabel('Accuracy')
plt.title('Accuracy per Test Size for KNN Classifier')
plt.xticks(test_sizes)
plt.grid(axis='y')
plt.show()