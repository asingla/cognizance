import numpy as np
from sklearn import preprocessing, cross_validation, svm, datasets, tree
import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv('voice.csv')

print(df.head())

X = df.drop(['label'], 1)
Z = df['label']
con = preprocessing.LabelEncoder()

Z = con.fit_transform(Z)
X_train, X_test, Z_train, Z_test = cross_validation.train_test_split(X, Z, test_size = 0.2)

clf = svm.SVC(kernel = 'linear')

clf.fit(X_train, Z_train)

print(clf.score(X_test, Z_test))

model = LinearRegression()
model.fit(X_train, Z_train)
accu = model.score(X_test, Z_test)
print(accu)
