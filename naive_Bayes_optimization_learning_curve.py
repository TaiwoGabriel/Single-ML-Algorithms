import warnings
warnings.filterwarnings('ignore')


from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from pandas.plotting import scatter_matrix
from mlxtend.plotting import plot_learning_curves

"""
# Printing all data columns
desired_width=1024
pd.set_option('display.width', desired_width)
np.set_printoption(linewidth=desired_width)
pd.set_option('display.max_columns',20)
"""

# Load data
df = load_breast_cancer()

# Exploratory data analysis
data = pd.DataFrame(df['data'], columns=df['feature_names'])
data['class'] = df['target']
#print(data.to_string())  # Note: to_string() print the whole dataframe in the output
print('The shape of the data is', data.shape)
print(data.head(5).to_string())
print()
print(data.describe())

# Group data by class
grp_data = data.groupby(by='class')
print(grp_data)

# handling missing values in the data
#data = data.isnull().sum()
#print('Missing values in each feature \n')
#print(data)


# Visualize the data
# Histrogram
data.hist() # plot the histograms of all features or variable in the data
plt.show()
# Note: In the result, The shape of the each graph can be Gaussian’, skewed or even has an exponential distribution.
# E.g, for instance area error, concavity error, fractal dimension error have exponential distribution,
# features like mean smoothness, worst smoothness, mean symmetry, worst texture have Gaussian or almost Gaussian
# distribution
# Finally, features like worst symmetry, worst radius, mean concave point, worst concave point have skewed distribution

# Density Plot
data.plot(kind='density', subplots=True, layout=(6,6), sharex=False)
plt.show()


# Box and Whisker Plot
#data.plot(kind='box', subplots=True, layout=(6,6), sharex=False, sharey=False)
#plt.show()


# Scatter Plot
#scatter_matrix(data)
#plt.show()

# Correlation Matrix Plot
correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df['feature_names'])
ax.set_yticklabels(df['feature_names'])
plt.show()

# Inspecting each feature of the data
plt.eventplot(data['mean radius'])
plt.title('Mean Radius Column of Breast Cancer Data')
plt.show()

plt.hist(data['class'])
plt.title('Class Visualization')
plt.show()

plt.hist(data['mean radius'])
plt.title('Mean Radius Column of Breast Cancer Data')
plt.show()

plt.boxplot(data['mean radius'])
plt.title('Mean Radius Column of Breast Cancer Data')
plt.show()

plt.scatter(data['mean radius'], data['mean texture'])
plt.title('Mean Radius Column of Breast Cancer Data')
plt.xlabel('Mean Texture')
plt.ylabel('Mean Radius')
plt.show()



# Separating class and target
X = df.data
y = df.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# feature scaling
prep = StandardScaler()
X_train = prep.fit_transform(X_train)
X_test = prep.transform(X_test)


# Hyperparameter Optimization using GridSearch
NB = GaussianNB()
cv_method = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

# Create a dictionary of naive bayes parameters
params_NB = {'var_smoothing': np.logspace(0,-9,num=100) }
# var_smoothing indicates the laplace correction
# Also, priors represents the prior probabilities of the classes. If we specify this parameter while
# fitting the data, then the prior probabilities will not be justified according to the data

# Computing the GridSearch
NB_grid = RandomizedSearchCV(NB,params_NB,scoring='accuracy',cv=cv_method)

# Fitting the NB_grid
NB_grid.fit(X_train,y_train)

# Print the best parameter values
print('Best Parameter Values:', NB_grid.best_params_)
data = pd.DataFrame(NB_grid.cv_results_['params'])
data['Mean Test Scores'] = NB_grid.cv_results_['mean_test_score']
#data['CV Score'] = NB_grid.cv_results_['param_var_smoothing']
print(data)
print('Mean of Cross Validation',NB_grid.best_score_)
print()

# Predicting on train sample
y_pred1 = NB_grid.predict(X_train)
train_acc = accuracy_score(y_train,y_pred1)
print('The model accuracy on the training data = %.3f' % train_acc)

# Predicting on test sample
y_pred = NB_grid.predict(X_test)
test_acc = accuracy_score(y_test,y_pred)
#print(y_test)
#print()
#print(y_pred)
print('The model accuracy on the test data = %.3f' % test_acc)


plot_learning_curves(X_train,y_train,X_test,y_test,NB_grid,scoring='misclassification error')
plt.show()


# Another way to plot the Learning curve
# Obtain scores from learning curve function
# cv is the number of folds while performing Cross Validation
sizes, training_scores, testing_scores = learning_curve(NB_grid, X, y, cv=cv_method,
                                                        scoring='accuracy', train_sizes=np.linspace(0.01, 1.0, 50))

# Mean and Standard Deviation of training scores
mean_training = np.mean(training_scores, axis=1)
Standard_Deviation_training = np.std(training_scores, axis=1)

# Mean and Standard Deviation of testing scores
mean_testing = np.mean(testing_scores, axis=1)
Standard_Deviation_testing = np.std(testing_scores, axis=1)

# dotted blue line is for training scores and green line is for cross-validation score
plt.plot(sizes, mean_training, '--', color="b", label="Training score")
plt.plot(sizes, mean_testing, color="g", label="Cross-validation score")

# Drawing plot
plt.title("LEARNING CURVE FOR NB Classifier")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

