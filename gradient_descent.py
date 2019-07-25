import numpy as np


def gs_test(x1,x2):
    a = x1 + x2
    return a




def compute_error(x,y,m,b):
    pass



def run_gd(x,y,l_rate, n_iterations):
    np.random.seed(42)
    initial_m = np.random.rand()
    initial_b = np.random.rand()

    print("GD starting at m= {}, b= {}".format(initial_m,initial_b))
    print("Initial error= {}".format(compute_error(x,y,initial_m,initial_b))

run_gd(1,3,4,5)
