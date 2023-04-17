# -------------------------------------------------------------------------
# AUTHOR: Joshua Furman
# FILENAME: deep_learning.py
# SPECIFICATION: train and test multiple deep neural networks using Keras and a function named "build_model()" to define their architecture.
                    # The best model's weights, architecture, and learning curves will be printed, and TensorFlow must be installed using the provided command.
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# importing the libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def build_model(n_hidden, n_neurons_hidden, n_neurons_output, learning_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons_hidden, activation="relu"))
    model.add(keras.layers.Dense(n_neurons_output, activation="softmax"))
    opt = keras.optimizers.SGD(learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


# Using Keras to Load the Dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Creating a Validation Set and Scaling the Features
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

n_hidden = [2, 5, 10]
n_neurons = [10, 50, 100]
l_rate = [0.01, 0.05, 0.1]

highest_acc = 0

for h in n_hidden:
    for n in n_neurons:
        for l in l_rate:
            model = build_model(h, n, 10, l)
            history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))
            _, acc = model.evaluate(X_test, y_test)
            if acc > highest_acc:
                highest_acc = acc
                best_model = model
                best_params = "Number of Hidden Layers: " + str(h) + ", number of neurons: " + str(
                    n) + ", learning rate: " + str(l)

print("Highest accuracy so far:", highest_acc)
print("Best parameters:", best_params)

# Print the summary of the best model found
print(best_model.summary())

# Plotting the learning curves of the best model
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
