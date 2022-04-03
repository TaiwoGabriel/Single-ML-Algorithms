from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt



digits = load_digits()
X, y = digits.data, digits.target

#Checking  Data in dataframe
df = pd.DataFrame(X)
print(df)
print(df.head(5))
print()
df2 = df.shape
print(df2)
df3 = df.describe()
print(df3)

#Splitting data into training and test sets
X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=0.3, random_state=0)


#Visualize data
plt.gray() #doctest: +SKIP
plt.matshow(digits.images[9]) #doctest: +SKIP
plt.show() #doctest: +SKIP

#inspecting data
print(digits.keys())
print(digits['DESCR'])
print(digits['target_names'])


#Data preprocessing
scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_tr) # It is good to scale the training set only
X_tr = scaling.transform(X_tr)                          # and apply the scaled data to the test set
X_t = scaling.transform(X_t)


#model building
nn = MLPClassifier(hidden_layer_sizes=(200,), activation='relu', solver='adam', shuffle=True, tol=1e-4, random_state=1)
cv = cross_val_score(nn, X_tr, y_tr, cv=10)
test_score = nn.fit(X_tr, y_tr).score(X_t, y_t)
print('CV accuracy score: %0.3f' % np.mean(cv))
print('Test accuracy score: %0.3f' % (test_score))

