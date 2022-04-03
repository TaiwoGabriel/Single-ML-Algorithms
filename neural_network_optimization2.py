# Import Libraries

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
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
MLP_clasf = MLPClassifier()
cv_method = RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42)

# Create a dictionary of MLP hyperparameters

params_MLP = {'max_iter':np.linspace(100,1000),'hidden_layer_sizes': [(50,50,50)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam','lbfgs'],
    'alpha': np.linspace(0.0001, 0.05),
    'learning_rate': ['constant','adaptive']}

"""
# Using GridSearch to explore the best parameter for the a MLP model
MLP_Grid = GridSearchCV(estimator=MLP_clasf,param_grid=params_MLP,
                       scoring='accuracy',
                       cv=cv_method)
"""
# Using Random Search to explore the best parameter for the a MLP model
MLP_Grid = RandomizedSearchCV(MLP_clasf,params_MLP,scoring='accuracy',cv=cv_method)

# Fitting the parameterized model
MLP_Grid.fit(X_train,y_train)

# Print the best parameter values
print('Best Parameter Values:', MLP_Grid.best_params_)
data = pd.DataFrame(MLP_Grid.cv_results_['params'])
data['Mean Test Scores'] = MLP_Grid.cv_results_['mean_test_score']
print(data)
print('Mean of Cross Validation',MLP_Grid.best_score_)
print()

# Evaluate the optimized MLP model
y_pred = MLP_Grid.predict(X_test)
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
plot_confusion_matrix(MLP_Grid,X_test,y_test,display_labels=class_names)
plt.title('Confusion Matrix for MLP with the best optimized parameters')
plt.show()

# Training and Testing scores
train_score = MLP_Grid.score(X_train,y_train)
print('Train Score = %.2f' % train_score)
test_score = MLP_Grid.score(X_test,y_test) # Note test score is also the same as the accuracy score
print('The test score= %.2f' % test_score)

MSE = mean_squared_error(y_test,y_pred)
print('The mean squared error of MLP with the best optimized parameters: %.2f' % MSE)


# Estimating the bias-variance errors
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(MLP_Grid,X_train,y_train
                                                                 ,X_test,y_test,loss='mse',num_rounds=200,
                                                                 random_seed=20)
# Summary of Results
print('Average Expected Squared loss for MLP %.2f' % avg_expected_loss)
print('Average Expected Squared loss Bias error for MLP %.2f' % avg_bias)
print('Average Expected Squared loss Variance error for MLP %.2f' % avg_variance)
print()

avg_expected_loss2, avg_bias2, avg_variance2 = bias_variance_decomp(MLP_Grid,X_train,y_train
                                                                 ,X_test,y_test,loss='0-1_loss',
                                                                 random_seed=20, num_rounds=200)
# Summary of Results
print('Average Expected 0-1 loss for MLP %.2f' % avg_expected_loss2)
print('Average Expected 0-1 Bias error for MLP %.2f' % avg_bias2)
print('Average Expected 0-1 Variance error for MLP %.2f' % avg_variance2)


# Plot MLP learning curve
plot_learning_curves(X_train,y_train,X_test,y_test,MLP_Grid,scoring='misclassification error',style='ggplot',
                     print_model=True,legend_loc='best')
plt.title('Learning Curve for MLP with the best optimized parameters')
plt.xlabel("Training Set Size"),
plt.ylabel("Accuracy Score"),
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# To compute the auc score and plot roc_curve, you need to compute the probabilities for test samples
probs = MLP_Grid.predict_proba(X_test)
# keep probabilities for the positive outcome only
MLP_probs = probs[:, 1]


# Compute the false positive rate (false alarm rate) and true positive rate to draw the roc curve
fpr, tpr, threshold = roc_curve(y_test,MLP_probs)

# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.', label='Neural Network')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# Plot Title
plt.title('ROC Curve for MLP with the best optimized parameters')
# show the plot
plt.show()

# Compute the roc_auc_score
MLP_auc = roc_auc_score(y_test, MLP_probs)
print('The ROC AUC score %.2f' % MLP_auc)


# Another way to plot the Learning curve
# Obtain scores from learning curve function
# cv is the number of folds while performing Cross Validation
train_sizes, training_scores, testing_scores = learning_curve(MLP_Grid, X_train,y_train, cv=cv_method,
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
plt.title("LEARNING CURVE FOR MLP Classifier")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

