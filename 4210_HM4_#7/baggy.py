# -------------------------------------------------------------------------
# AUTHOR: Joshua Furman
# FILENAME: bagging_random_forest.py
# SPECIFICATION: reads a file of training instances of handwritten digits and uses decision trees,
                    # an ensemble classifier, and a Random Forest classifier to recognize those digits, with the accuracy tested using a separate file of samples.
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to
# work here only with standard vectors and arrays

# importing some Python libraries

#A huge problem I keep having is that the base classifier accuracy and Random Forest accuracy are continuously 0. I cannot figure the solution out
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier

dbTraining = []
dbTest = []
X_training = []
y_training = []
classVotes = []

with open('optdigits.tra', 'r') as file:
    for line in file:
        row = line.split(',')
        dbTraining.append(row)

with open('optdigits.tes', 'r') as file:
    for line in file:
        row = line.split(',')
        dbTest.append(row)

for i in range(len(dbTest)):
    classVotes.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

accuracy = 0

print("Started my base and ensemble classifier ...")

for k in range(20):
    X_training = []
    y_training = []

    bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

    for row in bootstrapSample:
        X_training.append(row[:-1])
        y_training.append(row[-1])

    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)
    clf = clf.fit(X_training, y_training)

    for i in range(len(dbTest)):
        testSample = dbTest[i]
        prediction = clf.predict([testSample[:-1]])[0]
        classVotes[i][int(prediction)] += 1

        if k == 0 and prediction == int(testSample[-1]):
            accuracy += 1

accuracy /= len(dbTest)
print("Finished my base classifier (fast but relatively low accuracy) ...")
print("My base classifier accuracy: " + str(accuracy))
print("")

accuracy = 0
for i in range(len(dbTest)):
    testSample = dbTest[i]
    max_votes_index = classVotes[i].index(max(classVotes[i]))
    if max_votes_index == int(testSample[-1]):
        accuracy += 1

accuracy /= len(dbTest)
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy))
print("")

print("Started Random Forest algorithm ...")

clf = RandomForestClassifier(n_estimators=20)
clf.fit(X_training, y_training)

accuracy = 0
for i in range(len(dbTest)):
    testSample = dbTest[i]
    class_predicted_rf = clf.predict([testSample[:-1]])[0]
    if class_predicted_rf == int(testSample[-1]):
        accuracy += 1

accuracy /= len(dbTest)
print("Random Forest accuracy: " + str(accuracy))
print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")

