import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
# given an array, return the 0th, 50th and 100th index and then return it when using np.delete()
test_idx = [0, 50, 100] 

# training data
train_target = np.delete(iris.target, test_idx) # corresponding labels for the training data
# array of 4D feature vectors
train_data = np.delete(iris.data, test_idx, axis=0) 

# testing data
test_target = iris.target[test_idx] # correct labels for the three test data points i.e. [0, 1, 2]
test_data = iris.data[test_idx] # the three test data points' feature vectors 

clf = tree.DecisionTreeClassifier() 
clf.fit(train_data, train_target)
print("target =", test_target)
print("prediction =", clf.predict(test_data))