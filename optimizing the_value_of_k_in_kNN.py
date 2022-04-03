# Optimizing the value of k in kNN

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from mlxtend.plotting import plot_learning_curves


# Load data
df = load_breast_cancer()

# Separating class and target
X = df.data
y = df.target


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
prep = MinMaxScaler()
X_train = prep.fit_transform(X_train)
X_test = prep.transform(X_test)

# define lists to collect scores
train_scores, test_scores = list(), list()

# Manually define the k values
values = [i for i in [1,3,5,7,9]]
# evaluate a naive Bayes Classifier for each smoothing values
for i in values:
    # configure the model
    KNN = KNeighborsClassifier(n_neighbors=i)
    # fit model on the training dataset
    KNN.fit(X_train, y_train)
    # evaluate on the train dataset
    train_yhat = KNN.predict(X_train)
    train_acc = accuracy_score(y_train, train_yhat)
    train_scores.append(train_acc)
    # evaluate on the test dataset
    test_yhat = KNN.predict(X_test)
    test_acc = accuracy_score(y_test, test_yhat)
    test_scores.append(test_acc)
    # summarize progress
    print('train: %.3f, test: %.3f' % (train_acc, test_acc))

# plot of train and test scores vs var smoothing values
plt.plot(values, train_scores, '-o', label='Train')
plt.plot(values, test_scores, '-o', label='Test')
plt.xlabel('')
plt.legend()
plt.show()
