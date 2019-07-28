import numpy as np

np.random.seed(42)

class lin_reg_GD():
    def __init__(self,x,y,iter,l_rate):
        self.x = np.c_[x,np.ones((x.shape[0],1))]
        self.y = y.reshape(-1,1)
        self.iter = iter
        self.l_rate = l_rate
        self.w = np.random.rand(self.x.shape[1],1)
        self.instances = len(self.x)
        self._compute_error(self.w)
        self._opt_weights()

    def _compute_error(self, w):
        error = 0
        for i in range(0,self.instances):
            Xi = self.x[i,:].reshape(-1,1)
            Yi = self.y[i,:]
            Xwi = np.dot((self.w).T,Xi)
            error += (Xwi-Yi)**2
        return np.sqrt(error / self.instances)

    def _opt_weights(self):
        for i in range(self.iter):
            Xw = np.dot(self.x,self.w)
            XwY = Xw - self.y
            w_gradients = (2*(np.dot(self.x.T, XwY)))/self.instances
            self.w = self.w - (self.l_rate * w_gradients)
        final_ws = self.w
        return final_ws, w_gradients.round(decimals=4)
