# Import Libraries

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,plot_confusion_matrix
from mlxtend.evaluate import bias_variance_decomp
from mlxtend.plotting import plot_learning_curves
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_auc_score,roc_curve
from sklearn.metrics import mean_absolute_error


# Load data
df = load_breast_cancer()

# Separate features and labels
X = df.data
y = df.target

#Split data
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.33, random_state=42)

# Normalize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#y = np.array(y).reshape(-1,1)
# For the label (y) transformation, it is expecting 2D array, got 1D array instead because y has 1-dimensional row
# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or
# array.reshape(1, -1) if it contains a single sample.


# Train the MLP Classifier using default hyperparameters
MLP_clasf = MLPClassifier(hidden_layer_sizes=(150,100,100),activation='relu',solver='lbfgs',
                          batch_size='auto',learning_rate='constant',max_iter=500,
                          shuffle=True,learning_rate_init=0.001,random_state=42, alpha=0.001)
                          # max_iter means the maximum number of iteration or epoch.
                          # max_iter can also be used as a stopping criteria
                          # Hyperparameters to tune: hidden layers six, activation function, solver, alpha used as
                          # regularization term, max_iter, learning rate initialization.

# The solver for weight optimization.
# ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
# ‘sgd’ refers to stochastic gradient descent.
# ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
# Note: The default solver ‘adam’ works pretty well on relatively large datasets
# (with thousands of training samples or more) in terms of both training time and validation score.
# For small datasets, however, ‘lbfgs’ can converge faster and perform better

# ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
# ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
# ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
# ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

# Reflecting practical approaches to the problem of determining the optimal architecture
# of the network for a given task, the question about the values for three parameters,
# the number of hidden nodes (including the number of hidden layers), learning
# rate η , and momentum rate α , becomes very important.
# Usually the optimal architecture
# is determined experimentally, but some practical guidelines exist. If several networks
# with different numbers of hidden nodes give close results with respect to error
# criteria after the training, then the best network architecture is the one with smallest
# number of hidden nodes. Practically, that means starting the training process with networks
# that have a small number of hidden nodes, increasing this number, and then
# analyzing the resulting error in each case. If the error does not improve with the increasing
# number of hidden nodes, the latest analyzed network configuration can be selected
# as optimal. Optimal learning and momentum constants are also determined experimentally,
# but experience shows that the solution should be found with η about 0.1 and α
# about 0.5.
# No of hidden nodes = input nodes X 2/3 + output nodes

MLP_clasf.fit(X_train,y_train)
y_pred = MLP_clasf.predict(X_test)
print('Actual Labels:',y_test)
print('Predicted Labels:',y_pred)
acc = accuracy_score(y_test,y_pred)
print('MLP Accuracy is %.3f' % acc)


# Classification Report
class_report = classification_report(y_test,y_pred)
print(class_report)

# Confusion Matrix
conf = confusion_matrix(y_test,y_pred)
print(conf)
print('Number of misclassified samples out of the total samples\n', (X_test.shape[0], (y_test != y_pred).sum()))

# Plot Confusion Matrix
class_names = df['target_names']
plot_confusion_matrix(MLP_clasf,X_test,y_test,display_labels=class_names)
plt.title('Confusion Matrix for MLP')
plt.show()


# Training and Testing scores
train_score = MLP_clasf.score(X_train,y_train)
print('Train Score = %.2f' % train_score)
test_score = MLP_clasf.score(X_test,y_test) # Note test score is also the same as the accuracy score
print('The test score= %.2f' % test_score)

# Compute the mean squared error
MSE = mean_squared_error(y_test,y_pred)
print('The mean squared error of MLP is %.2f' % MSE)

# Computer the mean absolute error
MAE =mean_absolute_error(y_test, y_pred)
print('The mean absolute error: %.2f' % MAE)

# Estimating the bias-variance errors
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(MLP_clasf,X_train,y_train
                                                                 ,X_test,y_test,loss='mse',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected Squared loss for MLP %.2f' % avg_expected_loss)
print('Average Expected Squared loss Bias error for MLP %.2f' % avg_bias)
print('Average Expected Squared loss Variance error for MLP %.2f' % avg_variance)
print()

avg_expected_loss2, avg_bias2, avg_variance2 = bias_variance_decomp(MLP_clasf,X_train,y_train
                                                                 ,X_test,y_test,loss='0-1_loss',
                                                                 random_seed=20, num_rounds=200)
# Summary of Results
print('Average Expected 0-1 loss for MLP %.2f' % avg_expected_loss2)
print('Average Expected 0-1 Bias error for MLP %.2f' % avg_bias2)
print('Average Expected 0-1 Variance error for MLP %.2f' % avg_variance2)

# Plot MLP learning curve
plot_learning_curves(X_train,y_train,X_test,y_test,MLP_clasf,scoring='misclassification error',style='ggplot',
                     print_model=True,legend_loc='best')
plt.show()
plt.title('Learning Curve for Multilayer Perceptron')


# To compute the auc score and plot roc_curve, you need to compute the probabilities for test samples
probs = MLP_clasf.predict_proba(X_test)
# keep probabilities for the positive outcome only
MLP_probs = probs[:, 1]


# Compute the false positive rate (false alarm rate) and true positive rate to draw the roc curve
fpr, tpr, threshold = roc_curve(y_test,MLP_probs)

# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.', label='MLP')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# Plot Title
plt.title('ROC Curve for MLP')
# show the plot
plt.show()

# Compute the roc_auc_score
MLP_auc = roc_auc_score(y_test, MLP_probs)
print('The ROC AUC score %.2f' % MLP_auc)
