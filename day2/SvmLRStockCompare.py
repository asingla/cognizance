import pandas as pd
import quandl, datetime
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')


df = df[['Adj. Open','Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)

X_late = X[-forecast_out:]
X = X[:-forecast_out]


df.dropna(inplace = True)

y = np.array(df['label'])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)


print('Confidence using Linear Regression')
clf = LinearRegression()

clf.fit(X_train, y_train)

accu = clf.score(X_test, y_test)
print(accu)
forecast_set = clf.predict(X_late)

##This gives the predicted values for the data we did not have.
#print(forecast_set)


## Creates a new column used for plotting, might take too much time to explpain, can show as it is
# df['Forecast'] = np.nan
#
# last_date = df.iloc[-1].name
# last_unix = last_date.timestamp()
# one_da = 86400
# next_unix = last_unix + one_da
#
# for i in forecast_set:
# 	next_date = datetime.datetime.fromtimestamp(next_unix)
# 	next_unix += one_da
# 	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
#
# print(df['Forecast'])

# df['Adj. Close'].plot()
# df['Forecast'].plot()
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()



print('Confidence using SVM')
clf = svm.SVR()

clf.fit(X_train, y_train)

accu = clf.score(X_test, y_test)
pred = clf.pred(X_late)

print(accu)
