# Import Libraries
import numpy as np
import  pandas as pd
from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.pyplot as plt


# Importing dataset
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target
cls_name = dataset['target_names']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
# Compare the performance of the experts
cv_method = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# Define the list of Classifiers
def get_models():
    models = dict()
    models['NB'] = GaussianNB()
    models['kNN'] = KNeighborsClassifier()
    models['DT'] = DecisionTreeClassifier()
    models['SVM'] = SVC(probability=True)
    models['MLP'] = MLPClassifier(hidden_layer_sizes=(100,100,100))
    return models

# Call the models
models = get_models()
# Cross validate the models
def evaluate_model(model, X_train, y_train):
    # evaluate the model and collect the results
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv_method, n_jobs=-1)
    return scores

# Get the list of cross validation scores to compare the models
results, names = list(), list()
print('Accuracy of each single expert:')
for name, model in models.items():
    # evaluate the model
    scores = evaluate_model(model, X_test, y_test)
    # store the results
    results.append(scores)
    names.append(name)
    # summarize the performance along the way
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()

print('kNN Classifier----------------------------------------------------------')
# kNN Classifier-----------------------------------------------------------------
kNN = KNeighborsClassifier(n_neighbors=5,p=2)
kNN.fit(X_train, y_train)

# Evaluating the KNN classifier model on test samples using accuracy and other metrics
y_pred = kNN.predict(X_test)
y_pred2 = kNN.predict(X_train)

acc1 = accuracy_score(y_test,y_pred)
print('KNN Accuracy on test score',acc1)

acc2 = accuracy_score(y_train,y_pred2)
print('kNN Accuracy on Train Set:',acc2)
# Confusion Matrix
conf = confusion_matrix(y_test,y_pred)
print('KNN Confusion Matrix is \n')
print(conf)

# Classification Report
clas_rpt = classification_report(y_test,y_pred)
print('KNN classification report: \n')
print(clas_rpt)

# predict probabilities for test sample
probs = kNN.predict_proba(X_test)
# keep probabilities for the positive outcome only
kNN_probs = probs[:, 1]
# Compute the false positive rate (false alarm rate) and true positive rate to draw the roc curve
fpr, tpr, threshold = roc_curve(y_test,kNN_probs)

# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.', label='kNN')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# Plot Title
plt.title('KNN ROC Curve')
# show the plot
plt.show()

# Compute the roc_auc_score
kNN_auc = roc_auc_score(y_test, kNN_probs)
print('KNN ROC AUC score %.2f' % kNN_auc)


# Estimating the bias-variance errors
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(kNN,X_train,y_train
                                                                 ,X_test,y_test,loss='mse',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected Squared loss for kNN %.2f' % avg_expected_loss)
print('Average Expected Squared loss for Bias error for kNN %.2f' % avg_bias)
print('Average Expected Squared loss for Variance error for kNN %.2f' % avg_variance)
print('\n')

print('naive Bayes-------------------------------------------------------------------')
# Naive Bayes Classifier-----------------------------------------------------------------
GNB = GaussianNB(var_smoothing=1e-9)
GNB.fit(X_train, y_train)

# Evaluating the GNB classifier model on test samples using accuracy and other metrics
y_pred = GNB.predict(X_test)
y_pred2 = GNB.predict(X_train)

acc1 = accuracy_score(y_test,y_pred)
print('naive Bayes Accuracy on test score:',acc1)

acc2 = accuracy_score(y_train,y_pred2)
print('naive Bayes Accuracy on Train Set:',acc2)
# Confusion Matrix
conf = confusion_matrix(y_test,y_pred)
print('naive Bayes Confusion Matrix:\n')
print(conf)

# Classification Report
clas_rpt = classification_report(y_test,y_pred)
print('NB classification report: \n')
print(clas_rpt)

# predict probabilities for test sample
probs = GNB.predict_proba(X_test)
# keep probabilities for the positive outcome only
GNB_probs = probs[:, 1]
# Compute the false positive rate (false alarm rate) and true positive rate to draw the roc curve
fpr, tpr, threshold = roc_curve(y_test,GNB_probs)

# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.', label='naive Bayes')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# Plot Title
plt.title('naive Bayes ROC Curve')
# show the plot
plt.show()

# Compute the roc_auc_score
GNB_auc = roc_auc_score(y_test, GNB_probs)
print('naive Bayes ROC AUC score %.2f' % GNB_auc)


# Estimating the bias-variance errors
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(GNB,X_train,y_train
                                                                 ,X_test,y_test,loss='mse',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected Squared loss for NB %.2f' % avg_expected_loss)
print('Average Expected Squared loss for Bias error for NB %.2f' % avg_bias)
print('Average Expected Squared loss for Variance error for NB %.2f' % avg_variance)
print('\n')



print('Decision Tree-------------------------------------------------------------------')
# Decision Tree Classifier-----------------------------------------------------------------
DT = DecisionTreeClassifier(criterion='gini',splitter='best',min_samples_split=2,min_samples_leaf=1)
DT.fit(X_train, y_train)

# Evaluating the DT classifier model on test samples using accuracy and other metrics
y_pred = DT.predict(X_test)
y_pred2 = DT.predict(X_train)

acc1 = accuracy_score(y_test,y_pred)
print('Decision Tree Accuracy on test score:',acc1)

acc2 = accuracy_score(y_train,y_pred2)
print('Decision Tree Accuracy on Train Set:',acc2)
# Confusion Matrix
conf = confusion_matrix(y_test,y_pred)
print('Decision Tree Confusion Matrix:\n')
print(conf)

# Classification Report
clas_rpt = classification_report(y_test,y_pred)
print('Decision Tree classification report: \n')
print(clas_rpt)

# predict probabilities for test sample
probs = DT.predict_proba(X_test)
# keep probabilities for the positive outcome only
DT_probs = probs[:, 1]
# Compute the false positive rate (false alarm rate) and true positive rate to draw the roc curve
fpr, tpr, threshold = roc_curve(y_test,DT_probs)

# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.', label='Decision Tree')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# Plot Title
plt.title('Decision Tree ROC Curve')
# show the plot
plt.show()

# Compute the roc_auc_score
DT_auc = roc_auc_score(y_test, DT_probs)
print('Decision Tree ROC AUC score %.2f' % DT_auc)


# Estimating the bias-variance errors
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(DT,X_train,y_train
                                                                 ,X_test,y_test,loss='mse',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected Squared loss for DT %.2f' % avg_expected_loss)
print('Average Expected Squared loss for Bias error for DT %.2f' % avg_bias)
print('Average Expected Squared loss for Variance error for DT %.2f' % avg_variance)
print('\n')



print('Support Vector Machines-------------------------------------------------------------------')
# SVM Classifier-----------------------------------------------------------------
SVC = SVC(kernel='rbf',C=1.0,gamma='scale',probability=True)
SVC.fit(X_train, y_train)

# Evaluating the SVC classifier model on test samples using accuracy and other metrics
y_pred = SVC.predict(X_test)
y_pred2 = SVC.predict(X_train)

acc1 = accuracy_score(y_test,y_pred)
print('SVC Accuracy on test score:',acc1)

acc2 = accuracy_score(y_train,y_pred2)
print('SVC Accuracy on Train Set:',acc2)
# Confusion Matrix
conf = confusion_matrix(y_test,y_pred)
print('SVC Confusion Matrix:\n')
print(conf)

# Classification Report
clas_rpt = classification_report(y_test,y_pred)
print('SVC classification report: \n')
print(clas_rpt)

# predict probabilities for test sample
probs = SVC.predict_proba(X_test)
# keep probabilities for the positive outcome only
SVC_probs = probs[:, 1]
# Compute the false positive rate (false alarm rate) and true positive rate to draw the roc curve
fpr, tpr, threshold = roc_curve(y_test,SVC_probs)

# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.', label='SVC')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# Plot Title
plt.title('SVC ROC Curve')
# show the plot
plt.show()

# Compute the roc_auc_score
SVC_auc = roc_auc_score(y_test, SVC_probs)
print('SVC ROC AUC score %.2f' % SVC_auc)


# Estimating the bias-variance errors
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(SVC,X_train,y_train
                                                                 ,X_test,y_test,loss='mse',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected Squared loss for SVC %.2f' % avg_expected_loss)
print('Average Expected Squared loss for Bias error for SVC %.2f' % avg_bias)
print('Average Expected Squared loss for Variance error for SVC %.2f' % avg_variance)
print('\n')



print('Neural Network-------------------------------------------------------------------')
# SVM Classifier-----------------------------------------------------------------
MLP = MLPClassifier(hidden_layer_sizes=(100,100,100),solver='sgd',activation='relu',max_iter=1000)
MLP.fit(X_train, y_train)

# Evaluating the MLP classifier model on test samples using accuracy and other metrics
y_pred = MLP.predict(X_test)
y_pred2 = MLP.predict(X_train)

acc1 = accuracy_score(y_test,y_pred)
print('MLP Accuracy on test score:',acc1)

acc2 = accuracy_score(y_train,y_pred2)
print('MLP Accuracy on Train Set:',acc2)
# Confusion Matrix
conf = confusion_matrix(y_test,y_pred)
print('MLP Confusion Matrix:\n')
print(conf)

# Classification Report
clas_rpt = classification_report(y_test,y_pred)
print('MLP classification report: \n')
print(clas_rpt)

# predict probabilities for test sample
probs = MLP.predict_proba(X_test)
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
plt.title('MLP ROC Curve')
# show the plot
plt.show()

# Compute the roc_auc_score
MLP_auc = roc_auc_score(y_test, MLP_probs)
print('MLP ROC AUC score %.2f' % MLP_auc)


# Estimating the bias-variance errors
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(MLP,X_train,y_train
                                                                 ,X_test,y_test,loss='mse',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected Squared loss for MLP %.2f' % avg_expected_loss)
print('Average Expected Squared loss for Bias error for MLP %.2f' % avg_bias)
print('Average Expected Squared loss for Variance error for MLP %.2f' % avg_variance)


