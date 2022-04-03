# Import Libraries

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load Dataset
df = fetch_openml('mnist_784', version=1)

# Separate feature vector from classes
X = df.data
y = df.target

print(X.shape)
print(y.shape)

# Split the data to training and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Call Classifier
MLP = MLPClassifier(max_iter=500, solver='sgd',hidden_layer_sizes=(784,100,2), random_state=1)

# Train the classifier
MLP.fit(X_train,y_train)

#Predict with the trained MLP on training and test sets
y_pred1 = MLP.predict(X_train)
y_pred2 = MLP.predict(X_test)

# Accuracy on training set
acc1 = accuracy_score(y_train,y_pred1)

# Accuracy on test set
acc2 = accuracy_score(y_test,y_pred2)

print('Accuracy on Training Set %.2f ' % acc1)
print('Accuracy on Training Set %.2f ' % acc2)
