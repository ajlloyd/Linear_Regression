import numpy as np
import traceback

##################

def gs_test(x1,x2):
    a = x1 + x2
    return a

from sklearn.datasets import make_regression

x,y = make_regression(n_samples=1000,n_features=20,n_informative=10,
                      n_targets=1,random_state=12)


#########################

def compute_error(x,y,weights):
    y = y.reshape(-1,1)
    n = len(x)

    error = 0
    for i in range(0,len(x)):
        xi=x[i,:].reshape(-1,1)
        yi=y[i,:]
        xw = np.dot(weights.T,xi)
        sq = (xw-yi)**2
        error += sq
    return float(np.sqrt(error / n))



def run_gd(x,y,l_rate, n_iterations):
    ones = np.ones((x.shape[0],1))
    x = np.c_[x,ones]
    initial_w=np.random.rand(x.shape[1],1)
    print("GD starting at initial weights...")
    print("Initial error = {}".format(compute_error(x,y,initial_w)))
    print("Optimising...")

    #compute_error(x,y,initial_m,initial_b)

run_gd(x,y,0.01,1000)
