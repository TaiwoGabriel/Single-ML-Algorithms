import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#import dataset
data = "C:/Users/Gabriel/Desktop/sonar_all_data.csv"
df = pd.read_csv(data,delimiter=',',header=None)
print(df.head())
# Check data shape
print(df.shape)
# Statistical Summary of dataset
print(df.describe())
# Check data info
#print(df.info())

# Check available labels and label distribution
class_labels = df[60].unique()
print(class_labels)
label_count = df[60].value_counts()
print(label_count)

#Missing values
miss_values = df.isnull().sum()
print(miss_values)

sns.histplot(df, x=df[0], y=df[2])
#plt.show()

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Convert the Dataframe to Numpy Arrays
X = X.values
y = y.values

#print(X)
#print(y)

# Encoding attributes or Label Encoding: Transform the labels Mine 'M' or Rock 'R' to numeric binary values
# Mine M = 1, Rock R = 0
enc = LabelEncoder()
y = enc.fit_transform(y)
#print(y)

#Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Scale feature values to
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

#Build neural nework model
MLP1 = MLPClassifier(hidden_layer_sizes=(25,25,25),activation="relu",solver='adam',
                    learning_rate="adaptive",learning_rate_init=0.1, max_iter=1000)

MLP1.fit(X_train,y_train)
y_pred = MLP1.predict(X_test)
model_accuracy = accuracy_score(y_test,y_pred)
print("MLP1 accuracy is {}".format(model_accuracy))

print("\n")
#Build neural nework model
MLP2 = MLPClassifier(hidden_layer_sizes=(50,25,25),activation="relu",solver='sgd',
                    learning_rate="constant",learning_rate_init=0.001, max_iter=1000)

MLP2.fit(X_train,y_train)
y_pred = MLP2.predict(X_test)
model_accuracy = accuracy_score(y_test,y_pred)
print("MLP2 accuracy is {}".format(model_accuracy))

print("\n")
#Build neural nework model
MLP3 = MLPClassifier(hidden_layer_sizes=(50,25,50),activation="tanh",solver='lbfgs',
                    learning_rate="adaptive",learning_rate_init=0.0001, max_iter=1000)

MLP3.fit(X_train,y_train)
y_pred = MLP3.predict(X_test)
model_accuracy = accuracy_score(y_test,y_pred)
print("MLP3 is {}".format(model_accuracy))

print("\n")
MLP4 = MLPClassifier(hidden_layer_sizes=(50,50,50),activation="logistic",solver='sgd',
                    learning_rate="constant",learning_rate_init=0.01, max_iter=1000)

MLP4.fit(X_train,y_train)
y_pred = MLP4.predict(X_test)
model_accuracy = accuracy_score(y_test,y_pred)
print("MLP4 accuracy is {}".format(model_accuracy))

print("\n")
MLP5 = MLPClassifier(hidden_layer_sizes=(50,50,25),activation="tanh",solver='adam',
                    learning_rate="adaptive",learning_rate_init=0.00001, max_iter=1000)

MLP5.fit(X_train,y_train)
y_pred = MLP5.predict(X_test)
model_accuracy = accuracy_score(y_test,y_pred)
print("MLP5 accuracy is {}".format(model_accuracy))
