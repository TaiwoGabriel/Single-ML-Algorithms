# kNN model development

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer,load_boston
from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,plot_confusion_matrix
from mlxtend.plotting import plot_learning_curves
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score


b_cancer = load_breast_cancer()
X = b_cancer.data
y = b_cancer.target
cls_name = b_cancer['target_names']



# Data preprocessing- Feature Scaling: It is important to scale features in a data because it is important for
# regression algorithms and algorithms using Euclidean distances ( e.g kNN, PCA or k-Means). Feature scaling is
# important for these algorithms because they are sensitive to the variation in magnitude and range across features.
# It further helps in speed-up the model training and prediction time. We can skip feature
# scaling in case of the tree-based models such as Decision Tree or probability-based models such as Naive Bayes
# because in these models, the weight is assigned according to the values present in the data
# Preprocess the whole data except target
data_prep = MinMaxScaler()
X = data_prep.fit_transform(X)

# NOTE: fit_transform method is used for the training dataset and transform method to the testing dataset,
# the reason being that we learn the scaling parameter from the training dataset, and used the same
# parameters to scale the testing dataset.

# target is already encoded, but we need to reverse the labels
# so that malignant is the positive class
y = np.where(y==0, 1, 0)


# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Hyperparameter Optimization using GridSearch and RepeatedStratifiedKFold function
kNN = KNeighborsClassifier()

# Since the target labels have fewer malignant labels than benign, stratification ensures that the
# proportion of the two labels in both train and test sets are the same as the proportion in the full
# dataset in each cross-validation repetition
cv_method = RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42)

# Create a dictionary of KNN parameters for the grid search.
# Here, we will consider K values between 1 and 9 and  p  values of 1 (Manhattan), 2 (Euclidean), and 5 (Minkowski)
param_kNN = {'n_neighbors': [1,3,5,7,9], 'p': [1,2,5]} # Metric: Manhattan (p=1), Euclidean (p=2) or
# Minkowski (any p larger than 2). Technically, p=1 and p=2 are also Minkowski metrics

# Define the kNN model using gridsearch and optimize accuracy
kNN_grid = GridSearchCV(kNN,param_kNN,scoring='accuracy',cv=cv_method)
kNN_grid.fit(X_train,y_train)

# Print the best parameter values for KNN
print('Best Parameter values =',kNN_grid.best_params_)

# Printing mean of cross validation accuracy
print('Cross validation mean score:', kNN_grid.best_score_)

# To extract more cross-validation results, we can call gs.csv_results - a dictionary consisting
# of run details for each fold
CV_result_for_each_fold = kNN_grid.cv_results_['mean_test_score']
print(CV_result_for_each_fold)

# Visualize the hyperparameter tuning results from the cross-validation. We define a data frame by
# combining gs.cv_results_['params'] and gs.cv_results_['mean_test_score'].
# The gs.cv_results_['params'] is an array of hyperparameter combinations.
result_kNN = pd.DataFrame(kNN_grid.cv_results_['params'])
# Create two columns for mean test score of cross validation and the other column for metric
result_kNN['test score'] = kNN_grid.cv_results_['mean_test_score']
result_kNN['metric'] = result_kNN['p'].replace([1,2,5],['Manhattan','Euclidean','Minskowsi'])
print(result_kNN)

# Visualize the result
plt.style.use('ggplot')
for i in ['Manhattan','Euclidean','Minskowsi']:
    temp = result_kNN[result_kNN['metric']==i]
    plt.plot(temp['n_neighbors'], temp['test score'], marker='.', label=i)
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel("Mean CV Score")
plt.title("KNN Performance Comparison")
plt.show()

# Evaluating the KNN classifier model on test samples using accuracy and other metrics
y_pred = kNN_grid.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print('Model accuracy on test samples',acc)

"""
# Counting the positive and negative classes
cnt = np.count_nonzero(y_test==1)
cnt2 = np.count_nonzero(y_test==0)
print('Positive class', cnt)
print('Negative class', cnt2)
# The code below can also be used to print accuracy using the .score()
#acc = NN_1.score(X_test,y_test)
#print('Accuracy', acc)
"""

# Confusion Matrix
conf = confusion_matrix(y_test,y_pred)
print('Confusion Matrix is \n')
print(conf)
plot_confusion_matrix(kNN_grid,X_test,y_test,display_labels=cls_name,normalize='true')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
clas_rpt = classification_report(y_test,y_pred)
print('The classification report is \n')
print(clas_rpt)

# predict probabilities for test sample
probs = kNN_grid.predict_proba(X_test)
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
plt.title('ROC Curve')
# show the plot
plt.show()

# Compute the roc_auc_score
kNN_auc = roc_auc_score(y_test, kNN_probs)
print('The ROC AUC score %.2f' % kNN_auc)

# Plot kNN learning curve
plot_learning_curves(X_train,y_train,X_test,y_test, kNN_grid,scoring='misclassification error',style='ggplot',
                     print_model=True,legend_loc='best')
plt.title('kNN Learning Curve')
plt.show()

# Mean squared error
MSE = mean_squared_error(y_test,y_pred)
print('The mean squared error of kNN is %.2f' % MSE)

# Estimating the bias-variance errors
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(kNN_grid,X_train,y_train
                                                                 ,X_test,y_test,loss='mse',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected Squared loss for kNN %.2f' % avg_expected_loss)
print('Average Expected Squared loss for Bias error for kNN %.2f' % avg_bias)
print('Average Expected Squared loss for Variance error for kNN %.2f' % avg_variance)
print()

avg_expected_loss2, avg_bias2, avg_variance2 = bias_variance_decomp(kNN_grid,X_train,y_train
                                                                 ,X_test,y_test,loss='0-1_loss',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected 0-1 loss for kNN %.2f' % avg_expected_loss2)
print('Average Expected 0-1 Bias error for kNN %.2f' % avg_bias2)
print('Average Expected 0-1 Variance error for kNN %.2f' % avg_variance2)

