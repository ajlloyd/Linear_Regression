import gradient_descent as gd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

##### TEST-REGRESSION-DATA:
xs,ys = make_regression(n_samples=1000,n_features=1,
                        n_informative=1,noise = 50, n_targets=1,random_state=12)
x_train, x_test, y_train, y_test = train_test_split(xs,ys,test_size=0.33,random_state=12)



##### SCALER:
sc = gd.scaler()
scaled = sc.fit_transform(x_train)
scaled_t = sc.fit_transform(x_test)

##### BATCH-GD: ----------------------------------------------------------------
clf1 = gd.reg_BGD(scaled,y_train,1000,0.01)

# train + score:
y_pred1 = clf1._predict(scaled)
print("BatchGD train:", mean_squared_error(y_pred1,y_train))
# test + score:
yt_pred1 = clf1._predict(scaled_t)
print("BatchGD test:", mean_squared_error(yt_pred1,y_test))


##### STOCHASTIC-GD: -----------------------------------------------------------
clf2 = gd.reg_SGD(scaled,y_train,iter=2,l_rate=0.5)

# train + score:
y_pred2 = clf2._predict(scaled)
print("StochasticGD train:", mean_squared_error(y_pred2,y_train))
# test + score:
yt_pred2 = clf2._predict(scaled_t)
print("StochasticGD test:", mean_squared_error(yt_pred2,y_test))


##### RIDGE-GD: ----------------------------------------------------------------
clf3 = gd.ridge(scaled,y_train,iter=5, l_rate=0.5, alpha=-0.1)

# train + score:
y_pred3 = clf3._predict(scaled)
print("Ridge train:", mean_squared_error(y_pred3,y_train))
# test + score:
yt_pred3 = clf3._predict(scaled_t)
print("Ridge test:",mean_squared_error(yt_pred3,y_test))



##### PLOTS: -------------------------------------------------------------------
plt.plot(x_train, y_train, "r.")

line_x = np.linspace(-3,3,100)
clf1_ys = clf1._predict(line_x)
clf2_ys = clf2._predict(line_x)
clf3_ys = clf3._predict(line_x)

#plt.plot(line_x,clf1_ys)
plt.plot(line_x,clf2_ys, "b-")
plt.plot(line_x,clf3_ys,"g-")

plt.show()
