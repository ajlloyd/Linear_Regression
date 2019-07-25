import numpy as np


def gs_test(x1,x2):
    a = x1 + x2
    return a










def run_gd(x,y,l_rate, n_iterations):

    np.random.seed(42)
    initial_b = np.random.rand()
    initial_m = np.random.rand()
    print("GD starting at b= {}, m= {}".format(initial_b,initial_m))

run_gd(1,3,4,5)
