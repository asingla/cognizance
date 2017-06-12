import numpy as np
import matplotlib.pyplot as plt

def best_fit(X, Y):
	a1 = ((X*Y).mean() - X.mean()*Y.mean())/((X**2).mean() - (X.mean())**2)
	a0 = Y.mean() - a1*X.mean()
	return a0, a1

X = np.array([1,2,3,4,5,6])
Y = np.array([1.5,4,4,6, 7,8])

plt.scatter(X, Y)
a0,a1 = best_fit(X, Y)

X1 = 0
pred1 = a0 + a1 * X1
X2 = 10
pred2 = a0 + a1 * X2

X_Line = [X1, X2]
Y_Line = [pred1, pred2]
print(X_Line)
print(Y_Line)
plt.plot(X_Line, Y_Line)
plt.show()


