# Import Libraries

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,mean_absolute_error
from sklearn.metrics import confusion_matrix,classification_report,mean_absolute_error
from sklearn.model_selection import train_test_split,learning_curve
from mlxtend.evaluate import bias_variance_decomp
from mlxtend.plotting import plot_learning_curves
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Load dataset
df = load_breast_cancer()

# Separate Features from Classes
X = df.data
y = df.target

# Split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# call MLP classifier
mlp_gs = MLPClassifier(max_iter=1000)
parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam','lbfgs'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive']}

clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels

# Check best hyperparameters
print('Best parameters found:\n', clf.best_params_)

# Check all the scores for all combinations
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    y_true, y_pred = y_test, clf.predict(X_test)
    print('Results on the test set:')
    print(classification_report(y_true, y_pred))
