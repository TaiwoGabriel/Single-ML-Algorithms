# Import Libraries

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
from sklearn.model_selection import train_test_split,RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold
from mlxtend.evaluate import bias_variance_decomp
from mlxtend.plotting import plot_learning_curves
from sklearn.metrics import roc_auc_score,mean_squared_error,roc_curve
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Load data
df = load_breast_cancer()

# Separate features and classes from the data
X = df.data
y = df.target


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Scale Data
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


# Cross Validation Process
SVM_clasf = SVC(probability=True)
cv_method = RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42)

# Create a dictionary of SVM hyperparameters
# Parameter space for linear and rbf kernels
params_SVC = {'kernel':['linear','rbf','poly','sigmoid'],'C':np.linspace(0.1,1.0),
              'gamma':np.linspace(0.1,1.0),'degree':[1,2,3,4,5]}

"""
# Using GridSearch to explore the best parameter for the a SVM model
SVC_Grid = GridSearchCV(estimator=SVM_clasf,param_grid=params_SVC,
                       scoring='accuracy',
                       cv=cv_method)
"""
# Using Random Search to explore the best parameter for the a SVM model
SVC_Grid = RandomizedSearchCV(SVM_clasf,params_SVC,scoring='accuracy',cv=cv_method)


# Fitting the parameterized model
SVC_Grid.fit(X_train,y_train)

# Print the best parameter values
print('Best Parameter Values:', SVC_Grid.best_params_)
data = pd.DataFrame(SVC_Grid.cv_results_['params'])
data['Mean Test Scores'] = SVC_Grid.cv_results_['mean_test_score']
#data['CV Score'] = NB_grid.cv_results_['param_var_smoothing']
print(data)
print('Mean of Cross Validation',SVC_Grid.best_score_)
print()

# Evaluate the optimized SVM model
y_pred = SVC_Grid.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print('SVM Accuracy is %.3f' % acc)

# Classification Report
class_report = classification_report(y_test,y_pred)
print(class_report)

# Confusion Matrix
conf = confusion_matrix(y_test,y_pred)
print(conf)
print('Number of misclassified samples out of the total samples\n', (X_test.shape[0], (y_test != y_pred).sum()))

# Plot Confusion Matrix
class_names = df['target_names']
plot_confusion_matrix(SVC_Grid,X_test,y_test,display_labels=class_names)
plt.title('Confusion Matrix for SVM with the best optimized parameters')
plt.show()


# Training and Testing scores
train_score = SVC_Grid.score(X_train,y_train)
print('Train Score = %.2f' % train_score)
test_score = SVC_Grid.score(X_test,y_test) # Note test score is also the same as the accuracy score
print('The test score= %.2f' % test_score)

MSE = mean_squared_error(y_test,y_pred)
print('The mean squared error of SVM with the best optimized parameters: %.2f' % MSE)

# Estimating the bias-variance errors
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(SVC_Grid,X_train,y_train
                                                                 ,X_test,y_test,loss='mse',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected Squared loss for SVM %.2f' % avg_expected_loss)
print('Average Expected Squared loss Bias error for SVM %.2f' % avg_bias)
print('Average Expected Squared loss Variance error for SVM %.2f' % avg_variance)
print()

avg_expected_loss2, avg_bias2, avg_variance2 = bias_variance_decomp(SVC_Grid,X_train,y_train
                                                                 ,X_test,y_test,loss='0-1_loss',
                                                                 random_seed=20, num_rounds=200)
# Summary of Results
print('Average Expected 0-1 loss for SVM %.2f' % avg_expected_loss2)
print('Average Expected 0-1 Bias error for SVM %.2f' % avg_bias2)
print('Average Expected 0-1 Variance error for SVM %.2f' % avg_variance2)

# Plot SVM learning curve
plot_learning_curves(X_train,y_train,X_test,y_test,SVC_Grid,scoring='misclassification error',style='ggplot',
                     print_model=True,legend_loc='best')
plt.title('Learning Curve for SVM with the best optimized parameters')
plt.xlabel("Training Set Size"),
plt.ylabel("Accuracy Score"),
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# To compute the auc score and plot roc_curve, you need to compute the probabilities for test samples
probs = SVC_Grid.predict_proba(X_test)
# keep probabilities for the positive outcome only
Sup_vec_mach_probs = probs[:, 1]


# Compute the false positive rate (false alarm rate) and true positive rate to draw the roc curve
fpr, tpr, threshold = roc_curve(y_test,Sup_vec_mach_probs)

# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.', label='Support Vector Machines')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# Plot Title
plt.title('ROC Curve for SVM with the best optimized parameters')
# show the plot
plt.show()

# Compute the roc_auc_score
Sup_vec_mach_auc = roc_auc_score(y_test, Sup_vec_mach_probs)
print('The ROC AUC score %.2f' % Sup_vec_mach_auc)


# Another way to plot the Learning curve
# Obtain scores from learning curve function
# cv is the number of folds while performing Cross Validation
train_sizes, training_scores, testing_scores = learning_curve(SVC_Grid, X_train,y_train, cv=cv_method,
                                                        scoring='accuracy', train_sizes=np.linspace(0.01, 1.0, 10))

# Mean and Standard Deviation of training scores
mean_training = np.mean(training_scores, axis=1)
Standard_Deviation_training = np.std(training_scores, axis=1)

# Mean and Standard Deviation of testing scores
mean_testing = np.mean(testing_scores, axis=1)
Standard_Deviation_testing = np.std(testing_scores, axis=1)

# dotted blue line is for training scores and green line is for cross-validation score
plt.plot(train_sizes, mean_training, '--', color="b", label="Training score")
plt.plot(train_sizes, mean_testing, color="g", label="Testing score")

# Drawing plot
plt.title("LEARNING CURVE FOR SVM Classifier")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()
