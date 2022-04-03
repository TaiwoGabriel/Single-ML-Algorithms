Optimizing the performance of decision trees on breast cancer dataset

# Import Libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,plot_confusion_matrix
from mlxtend.evaluate import bias_variance_decomp
from mlxtend.plotting import plot_learning_curves
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_auc_score,roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import learning_curve


# Load data
df = load_breast_cancer()


# Separate features and labels
X = df.data
y = df.target

#Split data
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.33, random_state=42)

# Cross Validation Process
DT = DecisionTreeClassifier()
cv_method = RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42)

# Create a dictionary of DT hyperparameters
params_DT = {'criterion':['gini','entropy'],
             'max_depth':[1,2,3,4,5,6,7,8],
             'min_samples_split':[2,3],
             'splitter':['best','random']}

"""
# Using GridSearch to explore the best parameter for the a decision tree model
DT_Grid = GridSearchCV(estimator=DT,param_grid=params_DT,
                       scoring='accuracy',
                       cv=cv_method)
"""

# Using Random Search to explore the best parameter for the a decision tree model
DT_Grid = RandomizedSearchCV(DT,params_DT,scoring='accuracy',cv=cv_method)


# Fitting the parameterized model
DT_Grid.fit(X_train,y_train)

# Print the best parameter values
print('Best Parameter Values:', DT_Grid.best_params_)
data = pd.DataFrame(DT_Grid.cv_results_['params'])
data['Mean Test Scores'] = DT_Grid.cv_results_['mean_test_score']
#data['CV Score'] = NB_grid.cv_results_['param_var_smoothing']
print(data)
print('Mean of Cross Validation',DT_Grid.best_score_)
print()

# Evaluate the DT model on a test data
y_pred = DT_Grid.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print('Decision Tree accuracy on test data = %.2f' % acc)

# Training and Test score of the DT model
train_score = DT_Grid.score(X_train,y_train)
print('Training Score: %.2f' % train_score)
test_score = DT_Grid.score(X_test,y_test)
print('Test score: %.2f' % test_score)

# Confusion Matrix
conf = confusion_matrix(y_test,y_pred)
print(conf)
print('Number of misclassified samples out of the total samples\n', (X_test.shape[0], (y_test != y_pred).sum()))

# Plot Confusion Matrix
class_names = df['target_names']
plot_confusion_matrix(DT_Grid,X_test,y_test,display_labels=class_names)
plt.title('Confusion Matrix for Decision Tree')
plt.show()

# Classification Report
class_report = classification_report(y_test,y_pred)
print(class_report)

MSE = mean_squared_error(y_test,y_pred)
print('The mean squared error of DT is %.2f' % MSE)

# Estimating the bias-variance errors
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(DT_Grid,X_train,y_train
                                                                 ,X_test,y_test,loss='mse',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected Squared loss for DT %.2f' % avg_expected_loss)
print('Average Expected Squared loss Bias error for DT %.2f' % avg_bias)
print('Average Expected Squared loss Variance error for DT %.2f' % avg_variance)
print()

avg_expected_loss2, avg_bias2, avg_variance2 = bias_variance_decomp(DT_Grid,X_train,y_train
                                                                 ,X_test,y_test,loss='0-1_loss',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected 0-1 loss for DT %.2f' % avg_expected_loss2)
print('Average Expected 0-1 Bias error for DT %.2f' % avg_bias2)
print('Average Expected 0-1 Variance error for DT %.2f' % avg_variance2)

# Plot Decision Tree learning curve
plot_learning_curves(X_train,y_train,X_test,y_test,DT_Grid,scoring='misclassification error',style='ggplot',
                     print_model=True,legend_loc='best')
plt.show()


# To compute the auc score and plot roc_curve, you need to compute the probabilities for test samples
probs = DT_Grid.predict_proba(X_test)
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
plt.title('ROC Curve')
# show the plot
plt.show()


# Compute the roc_auc_score
DT_auc = roc_auc_score(y_test, DT_probs)
print('The ROC AUC score %.2f' % DT_auc)


"""
# Another way to plot the Learning curve
# Obtain scores from learning curve function
# cv is the number of folds while performing Cross Validation
sizes, training_scores, testing_scores = learning_curve(DT_Grid, X,y,cv=cv_method,
                                                        scoring='accuracy', train_sizes=np.linspace(0.01, 1.0, 50))

# Mean and Standard Deviation of training scores
mean_training = np.mean(training_scores, axis=1)
Standard_Deviation_training = np.std(training_scores, axis=1)

# Mean and Standard Deviation of testing scores
mean_testing = np.mean(testing_scores, axis=1)
Standard_Deviation_testing = np.std(testing_scores, axis=1)

# dotted blue line is for training scores and green line is for cross-validation score
plt.plot(sizes, mean_training, '--', color="b", label="Training score")
plt.plot(sizes, mean_testing, color="g", label="Testing score")

# Drawing plot
plt.title("LEARNING CURVE FOR DT Classifier")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()
"""
