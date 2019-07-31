import gradient_descent as gd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

np.set_printoptions(suppress=True)

##### TEST-REGRESSION-DATA:
xs,ys = make_regression(n_samples=10000,n_features=15,
                        n_informative=5,n_targets=1,random_state=12)
x_train, x_test, y_train, y_test = train_test_split(xs,ys,test_size=0.33,random_state=12)

##### SCALER:
sc = gd.scaler()
scaled = sc.fit_transform(x_train)

##### BATCH-GD:
clf1 = gd.reg_BGD(scaled,y_train,1000,0.01)
y_pred1 = clf1._predict(x_train)
print(clf1._score(y_pred1,y_train))
#print(np.sqrt(mean_squared_error(y_pred1,y_train)))

##### STOCHASTIC-GD:
clf2 = gd.reg_SGD(scaled,y_train,5,0.5)
y_pred2 = clf2._predict(x_train)
print(clf2._score(y_pred2,y_train))
#print(np.sqrt(mean_squared_error(y_pred2,y_train)))
