'''
Kyle Tranfaglia
COSC311 - Homework02
Last updated 03/25/24
This program reads in a food type dataset, randomly splits the dataset into two parts: 80% for training and 20% for testing,
and uses various classification algorithms in attempt to obtain the highest testing accuracy on the testing data. A classification Report
is used to show the classification performance and a heatmap is used to show the classification confusion matrix.
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

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

# KNN (K-Nearest Neighbors) Algorithm to fit the classifier to the training data and target labels and return the predictions for the test data
def knnClassifier(attributeTrain, attributeTest, targetTrain):
    knn = KNeighborsClassifier(n_neighbors = 1)  # Instantiate a KNeighborsClassifier object with 1 neighbors, optimal for data set
    knn.fit(attributeTrain, targetTrain)  # Fit the classifier to the training data and target labels

    return knn.predict(attributeTest)  # Return the predictions for the test data

# MLP (Multi-Layer Perceptron) Algorithm to fit the classifier to the training data and target labels and return the predictions for the test data
def mlpClassifier(attributeTrain, attributeTest, targetTrain):
    mlp = MLPClassifier(hidden_layer_sizes = 100, activation = 'tanh', solver = 'adam', alpha = 1e-5, batch_size = 36, tol = 1e-6, learning_rate_init=0.01,
                               learning_rate='constant', max_iter = 10000, random_state = 7)  # Instantiate an MLPClassifier object with optimized parameters
    mlp.fit(attributeTrain, targetTrain)  # Fit the classifier to the training data and target labels

    return mlp.predict(attributeTest)  # Return the predictions for the test data

# RF (Random Forest) Algorithm to fit the classifier to the training data and target labels and return the predictions for the test data
def rfClassifier(attributeTrain, attributeTest, targetTrain):
    rf = RandomForestClassifier(n_estimators = 315, criterion = 'gini', max_depth = 14, min_samples_split = 3,
                                        min_samples_leaf = 3, max_features = 'sqrt', random_state = 7)  # Instantiate an RFClassifier object with optimized parameters
    rf.fit(attributeTrain, targetTrain)  # Fit the classifier to the training data and target labels

    return rf.predict(attributeTest)  # Return the predictions for the test data

# SVC (Support Vector Classifier) Algorithm to fit the classifier to the training data and target labels and return the predictions for the test data
def svcClassifier(attributeTrain, attributeTest, targetTrain):
    svc = SVC(kernel = 'rbf', C = 13, gamma = 'scale', random_state = 7)  # Instantiate an SVCClassifier object with optimized parameters
    svc.fit(attributeTrain, targetTrain)  # Fit the classifier to the training data and target labels

    return svc.predict(attributeTest)  # Return the predictions for the test data

# Logistic Regression Algorithm to fit the classifier to the training data and target labels and return the predictions for the test data
def logRegClassifier(attributeTrain, attributeTest, targetTrain):
    logReg = LogisticRegression(solver='liblinear', random_state = 7)  # Instantiate an LogisticRegressionClassifier object with optimized parameters
    logReg.fit(attributeTrain, targetTrain)  # Fit the classifier to the training data and target labels

    return logReg.predict(attributeTest)  # Return the predictions for the test data

# Pipelined Logistic Regression and polynomial feature transformation Algorithm to fit the classifier to the training data and target labels and return predictions
def pipelineClassifier(attributeTrain, attributeTest, targetTrain):
    # Note: Increasing polynomial features count improves accuracy but significantly decreases performance
    # Instantiate a PiplinedClassifier object to combine polynomial feature transformation with logistic regression with optimized parameters
    pipeline = make_pipeline(PolynomialFeatures(2), LogisticRegression(solver='liblinear', random_state = 7))
    pipeline.fit(attributeTrain, targetTrain)  # Fit the classifier to the training data and target labels

    return pipeline.predict(attributeTest)  # Return the predictions for the test data

# Voting Algorithm to fit the classifier to the training data and target labels and return the predictions for the test data
# Algorithm uses all the other classifers and uses a weighted average of predicted probabilities for create a prediction
def votingClassifier(attributeTrain, attributeTest, targetTrain):
    # All Classifiers for voting
    knn = KNeighborsClassifier(n_neighbors = 1)
    mlp = MLPClassifier(hidden_layer_sizes = 100, activation = 'tanh', solver = 'adam', alpha = 1e-5, batch_size = 36, tol = 1e-6, learning_rate_init=0.01,
                               learning_rate='constant', max_iter = 10000, random_state = 7)
    rf = RandomForestClassifier(n_estimators = 315, criterion = 'gini', max_depth = 14, min_samples_split = 3,
                                        min_samples_leaf = 3, max_features = 'sqrt', random_state = 7)
    svc = SVC(kernel = 'rbf', C = 13, gamma = 'scale', probability = True, random_state = 7)
    pipeline = make_pipeline(PolynomialFeatures(2), LogisticRegression(solver='liblinear', random_state = 7))

    # Create a voting ensemble of the classifiers with soft voting. Weighted average of predicted probabilities
    votingSystem = VotingClassifier(estimators=[('mlp', mlp), ('knn', knn), ('svc', svc), ('rf', rf), ('pipeline', pipeline)], voting='soft')
    votingSystem.fit(attributeTrain, targetTrain)  # Fit the classifier to the training data and target labels

    return votingSystem.predict(attributeTest)  # Return the predictions for the test data

# Condense features by eliminating them based on importance
def condenseFeatures(attributeTrain, targetTrain, numFeatures):
    estimator = LogisticRegression(max_iter=1000)  # Instantiate logistic regression as the estimator

    # Instantiate RFE with logistic regression estimator and recursively select top _ features, then fit RFE to the data
    rfe = RFE(estimator, n_features_to_select = numFeatures)
    rfe.fit(attributeTrain, targetTrain)

    return rfe.support_  # Return selected features

# Print the confusion matrix, classification report, accuracy score, heatmap, and matrix display
def printResults(targetTest, prediction):
    confusionMatrix = confusion_matrix(targetTest, prediction)  # Create confusion matrix

    # Display the confusion matrix, a classification report, and the overall accuracy score of the prediction
    print("\nConfusion Matrix:\n", confusionMatrix)
    print("\nClassification Report:\n", classification_report(targetTest, prediction))
    print("Accuracy Score:", accuracy_score(targetTest, prediction))

    # Display a heatmap / confusion matrix using matplotlib and the sklearn toolset
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

# main
foodData = pd.read_csv('FoodTypeDataset.csv')  # Read in data form csv file

foodData.columns = ["column" + str(i + 1) for i in range(0, len(foodData.columns) - 1)] + ["Target"]  # Set up a list to store column labels (numbered columns)

attributes = foodData.iloc[:, :-1]  # Stores all the attributes, as in, every column except target
target = foodData.iloc[:, -1]  # Stores last column (Target)

# Declare train and test variables, then split the datasets into training and testing subsets
attributeTrain, attributeTest, targetTrain, targetTest = train_test_split(attributes, target, test_size = .2, shuffle = True, random_state = 7)

scaler = StandardScaler()  # Instantiate a StandardScaler object
scaler.fit(attributeTrain)  # Fit the scaler to the training data to compute the mean and standard deviation
# Scale the training data and testing data using the mean and standard deviation
attributeTrain = scaler.transform(attributeTrain)
attributeTest = scaler.transform(attributeTest)

# Get and Use only the top features for the data set, as in, the most correlated and informative features
# topFeatures = condenseFeatures(attributeTrain, targetTrain, 10)

# attributeTrain = attributeTrain[:, topFeatures]
# attributeTest = attributeTest[:, topFeatures] 

# Get predictions for classifiers and print the results
# prediction = knnClassifier(attributeTrain, attributeTest, targetTrain)
# printResults(targetTest, prediction)

# prediction = mlpClassifier(attributeTrain, attributeTest, targetTrain)
# printResults(targetTest, prediction)

# prediction = rfClassifier(attributeTrain, attributeTest, targetTrain)
# printResults(targetTest, prediction)

# prediction = svcClassifier(attributeTrain, attributeTest, targetTrain)
# printResults(targetTest, prediction)

# prediction = logRegClassifier(attributeTrain, attributeTest, targetTrain)
# printResults(targetTest, prediction)

# prediction = pipelineClassifier(attributeTrain, attributeTest, targetTrain)
# printResults(targetTest, prediction)

prediction = votingClassifier(attributeTrain, attributeTest, targetTrain)
printResults(targetTest, prediction)