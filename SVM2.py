
#Support vector machine classification
import numpy as np
from sklearn.svm import SVC
X = np.array([[-1, -1], [-2, -1], [1, 1], [-2, 1]]) #training data
y = np.array([1, 2, 1, 2]) #class labels

clf = SVC(kernel='linear')
clf.fit(X, y)
prediction = clf.predict([[0,6]]) #[0,6] is the test data
print(prediction)




#Another example of linear support vector classifier
import numpy as np
import matplotlib.pyplot as plt #used for data visualization
#from matplotlib import style
#style.use("ggplot")
from sklearn import svm
X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])
y = np.array([0,1,0,1,0,1])
print(X.view())
#clf = svm.SVC(kernel='linear', C = 1.0)
#clf.fit(X,y)
#result = clf.predict([0.58,0.76])
#print(result)



#the code below is used for plotting the graph of how SVC classifies the samples
w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()

