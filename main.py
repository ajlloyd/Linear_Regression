import gradient_descent as gd
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

np.set_printoptions(suppress=True)

xs,ys = make_regression(n_samples=10000,n_features=10,
                        n_informative=10,n_targets=1,random_state=12)

x_train, x_test, y_train, y_test = train_test_split(xs,ys,test_size=0.33,random_state=12)


sc = gd.scaler()
scaled = sc.fit_transform(x_train)


"""clf = LinearRegression()
a = clf.fit(scaled,y_train)
print(clf.coef_, clf.intercept_)"""


clf = gd.lin_reg_BGD(scaled,y_train,1000,0.01)

"""coefs = clf._coefficients()
print(coefs)

ints = clf._intercept()
print(ints)

grads = clf._gradients()
print(grads)"""

y_pred = clf._predict(x_train)

print(clf._score(y_pred,y_train))
print(mean_squared_error(y_pred, y_train.reshape(-1,1)))
