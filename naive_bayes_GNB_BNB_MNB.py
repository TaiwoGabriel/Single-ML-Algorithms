import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve
from mlxtend.evaluate import bias_variance_decomp
from mlxtend.plotting import plot_learning_curves
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Load in the data
df = load_iris()

# Inspecting the data in a DataFrame
data = pd.DataFrame(df['data'], columns=df['feature_names'])
data['class'] = df['target']
class_names = df['target_names']
#print(class_name)

X = df.data
y = df.target


# Split the random into 70% train set and 30% test set
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=1)


# Call GaussianNB
clf = GaussianNB(var_smoothing=1e-9) # NOTE: var_smoothing is used to solve zero probability problem
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# Computing the accuracy score
Acc_score = accuracy_score(y_test, y_pred)
print('The accuracy score of Gaussian NB is %.2f' % Acc_score)

# Computing the training and test scores
train_error = clf.score(X_train,y_train)
test_error = clf.score(X_test,y_test)
print('The training score is %.2f' % train_error)
print('The test score is %.2f' % test_error)

# Plot naive Bayes learning curve
plot_learning_curves(X_train,y_train,X_test,y_test,clf,scoring='misclassification error',style='ggplot',
                     print_model=True,legend_loc='best')
# Where to place the plot legend: {'best', 'upper left', 'upper right', 'lower left', 'lower right'}
# Note: scoring must be in 'misclassification error', 'accuracy', 'average_precision', 'f1', 'f1_micro',
# 'f1_macro', 'f1_weighted', 'f1_samples', 'log_loss', 'precision', 'recall', 'roc_auc', 'adjusted_rand_score',
# 'mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'r2']))
#plt.title('GaussianNB Learning Curve')
#plt.show()


# Computing the confusion matrix
conf = confusion_matrix(y_test,y_pred)
print('The confusion matrix of GaussianNB is', conf)
print('Number of misclassified samples out of the total samples\n', (X_test.shape[0], (y_test != y_pred).sum()))



# Plotting Confusion Matrix without Normalization
plot_confusion_matrix(clf,X_test,y_test,display_labels=class_names)
plt.title('Confusion Matrix without Normalization')
plt.show()

# Plotting Confusion Matrix with Normalization
plot_confusion_matrix(clf,X_test,y_test,display_labels=class_names,normalize='true') # Note: normalize must be one of
plt.title('Confusion Matrix with Normalization')                                   # {'true', 'pred', 'all', None}
plt.show()


# Classification Report
class_report = classification_report(y_test,y_pred)
print(class_report)

# Computing the probabilities of each test samples
#probab = clf.predict_proba(X_test)
#print('The probabilities of test samples are', probab)
#print('The predicted class is\n')
#print(y_pred)

# Mean squared error
MSE = mean_squared_error(y_test,y_pred)
print('The mean squared error of GaussianNB is %.2f' % MSE)

# Estimating the bias-variance errors
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(clf,X_train,y_train
                                                                 ,X_test,y_test,loss='mse',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected Squared loss for GNB %.2f' % avg_expected_loss)
print('Average Expected Squared loss Bias error for GNB %.2f' % avg_bias)
print('Average Expected Squared loss Variance error for GNB %.2f' % avg_variance)
print()

avg_expected_loss2, avg_bias2, avg_variance2 = bias_variance_decomp(clf,X_train,y_train
                                                                 ,X_test,y_test,loss='0-1_loss',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected 0-1 loss for GNB %.2f' % avg_expected_loss2)
print('Average Expected 0-1 Bias error for GNB %.2f' % avg_bias2)
print('Average Expected 0-1 Variance error for GNB %.2f' % avg_variance2)


"""
# Call MultinomialNB (MNB)
clf = MultinomialNB(alpha=1) # NOTE: alpha is used to solve zero-probability problem
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

Acc_score = accuracy_score(y_test, y_pred)
print('The accuracy score of MNB is %.2f' % Acc_score)

# Computing the training and test errors
train_error = clf.score(X_train,y_train)
test_error = clf.score(X_test,y_test)
print('The training score is %.2f' % train_error)
print('The test score is %.2f' % test_error)

# Plot naive Bayes learning curve
plot_learning_curves(X_train,y_train,X_test,y_test,clf,scoring='misclassification error',style='ggplot',
                     print_model=True,legend_loc='best')
plt.show()

conf = confusion_matrix(y_test,y_pred)
print('The confusion matrix of MNB is', conf)
print('Number of misclassified samples out of the total samples\n', (X_test.shape[0], (y_test != y_pred).sum()))

class_report = classification_report(y_test,y_pred)
print(class_report)

#probab = clf.predict_proba(X_test)
#print('The probabilities of test samples are', probab)
#print('The predicted class is\n')
#print(y_pred)

MSE = mean_squared_error(y_test,y_pred)
print('The mean squared error of MNB is %.2f' % MSE)

# Estimating the bias-variance errors
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(clf,X_train,y_train
                                                                 ,X_test,y_test,loss='mse',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected Squared loss for MNB %.2f' % avg_expected_loss)
print('Average Expected Squared loss Bias error for MNB %.2f' % avg_bias)
print('Average Expected Squared loss Variance error for MNB %.2f' % avg_variance)
print()

avg_expected_loss2, avg_bias2, avg_variance2 = bias_variance_decomp(clf,X_train,y_train
                                                                 ,X_test,y_test,loss='0-1_loss',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected 0-1 loss for MNB %.2f' % avg_expected_loss2)
print('Average Expected 0-1 Bias error for MNB %.2f' % avg_bias2)
print('Average Expected 0-1 Variance error for MNB %.2f' % avg_variance2)


# Call  BernoulliNB(BNB)
clf = BernoulliNB(alpha=1.0) # NOTE: alpha represent laplace smoothing to solve the zero probability problem
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

Acc_score = accuracy_score(y_test, y_pred)
print('The accuracy score of BNB is %.2f' % Acc_score)

conf = confusion_matrix(y_test,y_pred)
#precis = precision_score(y_test,y_pred,zero_division=1,average='weighted',pos_label=1) # Zero division value can be: "warn", 0, 1
#recal = recall_score(y_test,y_pred,zero_division=1,average='weighted',pos_label=1)
# NOTE: Target is multiclass but average='binary'.
# Please choose another average setting, one of [None, 'micro', 'macro', 'weighted']
#print('The precision score is %.2f' % precis)
#print('The recall score is %.2f' % recal)
print('The confusion matrix of BNB is', conf)
print('Number of misclassified samples out of the total samples\n', (X_test.shape[0], (y_test != y_pred).sum()))

class_report = classification_report(y_test,y_pred)
print(class_report)

# Computing the training and test errors
train_error = clf.score(X_train,y_train)
test_error = clf.score(X_test,y_test)
print('The training error is %.2f' % train_error)
print('The test error is %.2f' % test_error)

# Plot naive Bayes learning curve
plot_learning_curves(X_train,y_train,X_test,y_test,clf,scoring='misclassification error',style='ggplot',
                     print_model=True,legend_loc='best')
plt.show()

#probab = clf.predict_proba(X_test)
#print('The probabilities of test samples are', probab)
#print('The predicted class is\n')
#print(y_pred)

MSE = mean_squared_error(y_test,y_pred,multioutput='uniform_average') # NOTE: Allowed values for multioutput:
                                                       # 'raw_values', 'uniform_average', 'variance_weighted'
print('The mean squared error of BNB is %.2f' % MSE)

# Estimating the bias-variance errors
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(clf,X_train,y_train
                                                                 ,X_test,y_test,loss='mse',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected Squared loss for BNB %.2f' % avg_expected_loss)
print('Average Expected Squared loss Bias error for BNB %.2f' % avg_bias)
print('Average Expected Squared loss Variance error for BNB %.2f' % avg_variance)
print()

avg_expected_loss2, avg_bias2, avg_variance2 = bias_variance_decomp(clf,X_train,y_train
                                                                 ,X_test,y_test,loss='0-1_loss',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected 0-1 loss for BNB %.2f' % avg_expected_loss2)
print('Average Expected 0-1 Bias error for BNB %.2f' % avg_bias2)
print('Average Expected 0-1 Variance error for BNB %.2f' % avg_variance2)

"""
