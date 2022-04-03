# Import Libraries

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
#%matplotlib inline

# Load data
iris = load_iris()
class_name = iris['target_names']
#print(class_name)
irisdata = pd.DataFrame(iris['data'],columns=iris['feature_names'])
irisdata['Class'] = iris['target']
irisdata['Class'] = irisdata['Class'].replace([0,1,2],class_name)
print(irisdata)


#Visualize the data using seaborn pair plots
import seaborn as sns
sns.pairplot(irisdata,hue='Class',palette='Dark2')
plt.show()

# Split the data into feature vector and class
from sklearn.model_selection import train_test_split
X = irisdata.drop('Class', axis=1)
y = irisdata['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

# Apply kernels to transform the data to a higher dimension
kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']   # A function which returns the corresponding SVC model
def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernel
        return SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernel
        return SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return SVC(kernel='linear', gamma="auto")

# Train a SVC model using different kernel
for i in range(3):
    # Call the SVM classifier
    svclassifier = getClassifier(i)
    # Make prediction
    svclassifier.fit(X_train, y_train)
    # Evaluate our model
    y_pred = svclassifier.predict(X_test)
    print("Evaluation:", kernels[i], "kernel")
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
