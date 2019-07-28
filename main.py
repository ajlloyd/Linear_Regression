import gradient_descent as gd

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

xs,ys = make_regression(n_samples=10000,n_features=1,n_informative=1,n_targets=1,random_state=12)
sc = StandardScaler()
scaled = sc.fit_transform(xs)




opt = gd.lin_reg_GD(scaled,ys,1000,0.01)

optimised = opt._opt_weights()
print(optimised)
print(opt._compute_error(optimised))
