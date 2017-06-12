import numpy as np
from sklearn import datasets,preprocessing, cross_validation
from sklearn.linear_model import LinearRegression

dataset = datasets.load_boston()
X = dataset.data
Y = dataset.target
# separate training & testing data

print(X[0:2])
X = preprocessing.scale(X)
print(X[0:2])

X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(X, Y, test_size = 0.02) 

model = LinearRegression()
model.fit(X_Train, Y_Train)
b = model.predict(X_Test)
score = model.score(X_Test, Y_Test)
print(model.coef_)
print(score)
print(b)
print(Y_Test)
