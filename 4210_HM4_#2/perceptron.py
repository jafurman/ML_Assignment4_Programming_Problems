# -------------------------------------------------------------------------
# AUTHOR: Joshua Furman
# FILENAME: perceptron.py
# SPECIFICATION: Build a single layer perceptron and a multi level perceptron to classify handwritten digits
# FOR: CS 4210- Assignment #4
# TIME SPENT: 3 Hours
# -----------------------------------------------------------*/
# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

# Define the hyperparameters to iterate over
n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

# Read the training and test data
df = pd.read_csv('optdigits.tra', sep=',', header=None)
X_training = np.array(df.values)[:, :64]
y_training = np.array(df.values)[:, -1]

df = pd.read_csv('optdigits.tes', sep=',', header=None)
X_test = np.array(df.values)[:, :64]
y_test = np.array(df.values)[:, -1]

# Initialize the highest accuracy for both classifiers to 0
perceptron_highest_accuracy = 0
mlp_highest_accuracy = 0

# Iterate over learning rate, shuffle, and algorithm
for learning_rate in n:
    for shuffle in r:
        for algorithm in ['Perceptron', 'MLP']:
            # Create a Neural Network classifier
            if algorithm == 'Perceptron':
                clf = Perceptron(eta0=learning_rate, shuffle=shuffle, max_iter=3000)
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init=learning_rate, hidden_layer_sizes=(5,),
                                    shuffle=shuffle, max_iter=3000)

            # Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            # Make the classifier prediction for each test sample and start computing its accuracy
            highest_accuracy = 0
            for x_testSample, y_testSample in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])
                if prediction == y_testSample:
                    highest_accuracy += 1

            # Check if the calculated accuracy is higher than the previously one calculated for each classifier. If
            # so, update the highest accuracy and print it together with the network hyperparameters
            highest_accuracy /= len(y_test)
            if algorithm == 'Perceptron':
                if highest_accuracy > perceptron_highest_accuracy:
                    perceptron_highest_accuracy = highest_accuracy
                    perceptron_best_hyperparameters = {'learning_rate': learning_rate, 'shuffle': shuffle}
                    print(f"Highest Perceptron accuracy so far: {perceptron_highest_accuracy:.2f}, Parameters: {perceptron_best_hyperparameters}")
            else:
                if highest_accuracy > mlp_highest_accuracy:
                    mlp_highest_accuracy = highest_accuracy
                    mlp_best_hyperparameters = {'learning_rate': learning_rate, 'shuffle': shuffle}
                    print(f"Highest MLP accuracy so far: {mlp_highest_accuracy:.2f}, Parameters: {mlp_best_hyperparameters}")

