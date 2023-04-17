# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# importing necessary libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv
import numpy as np

# read the training data from csv file and populate dbTraining
with open('optdigits.tra', newline='') as csvfile:
    data = csv.reader(csvfile)
    dbTraining = [row for row in data]

# read the test data from csv file and populate dbTest
with open('optdigits.tes', newline='') as csvfile:
    data = csv.reader(csvfile)
    dbTest = [row for row in data]

# initialize class votes for each test sample
classVotes = [[0 for _ in range(10)] for _ in range(len(dbTest))]

print("Started my base and ensemble classifier ...")

accuracy = 0

# create 20 bootstrap samples and one classifier will be created for each bootstrap
for k in range(20):
    bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

    # populate X_training and y_training by using the bootstrapSample
    X_training = [row[:-1] for row in bootstrapSample]
    y_training = [row[-1] for row in bootstrapSample]

    # fit the decision tree to the data
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)
    clf = clf.fit(X_training, y_training)

    for i, testSample in enumerate(dbTest):
        # predict the class for each test sample and update the corresponding index value in classVotes
        predicted = clf.predict([testSample[:-1]])[0]
        classVotes[i][predicted] += 1

        if k == 0 and predicted == testSample[-1]:
            # calculate the accuracy of the base classifier for the first bootstrap sample
            accuracy += 1

    if k == 0:
        accuracy /= len(dbTest)
        print("Finished my base classifier (fast but relatively low accuracy) ...")
        print("My base classifier accuracy: " + str(accuracy) + "\n")

# calculate the accuracy of the ensemble classifier using the majority vote in classVotes
accuracy = 0

for i, testSample in enumerate(dbTest):
    predicted = np.argmax(classVotes[i])
    if predicted == testSample[-1]:
        accuracy += 1

accuracy /= len(dbTest)

# print the accuracy of the ensemble classifier
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy) + "\n")

print("Started Random Forest algorithm ...")

# create a Random Forest Classifier with 20 decision trees
clf = RandomForestClassifier(n_estimators=20)

# fit the Random Forest to the training data
X_training = [row[:-1] for row in dbTraining]
y_training = [row[-1] for row in dbTraining]
clf.fit(X_training, y_training)

# predict the class for each test sample and update the corresponding index value in classVotes
for i, testSample in enumerate(dbTest):
    predicted = clf.predict([testSample[:-1]])[0]
    classVotes[i][predicted] += 1

# calculate the accuracy of the Random Forest classifier
correct_rf = 0

for i in range(len(dbTest)):
    if np.argmax(classVotes[i]) == dbTest[i][-1]:
        correct_rf += 1

accuracy_rf = correct_rf / len(dbTest)

# print the accuracy of the Random Forest classifier
print("Random Forest accuracy: " + str(accuracy_rf) + "\n")
print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")


