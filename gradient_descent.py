import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

np.random.seed(42)

##### BATCH-GRADIENT-DESCENT:
class lin_reg_BGD():
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
        return final_ws.ravel(), w_gradients

    def _coefficients(self):
        return self._opt_weights()[0][:-1]

    def _intercept(self):
        return self._opt_weights()[0][-1]

    def _gradients(self):
        return self._opt_weights()[1]

    def _predict(self, new_x):
        opt_w = self._opt_weights()[0].reshape(-1,1)
        new_x_ones = np.c_[new_x,np.ones((new_x.shape[0],1))]
        return new_x_ones.dot(opt_w)

    def _score(self, y_pred, y):
        # sqrt((pred - actual)**2 / instances)
        #return np.sqrt(((y_pred - y)**2) / self.instances)
        error = 0
        for i in range(0,self.instances):
            YPi = y_pred[i,:].reshape(-1,1) #pred
            Yi = self.y[i,:] #true

            error += (YPi-Yi)**2
        return (error / self.instances)







##### STANDARD-SCALER:
class scaler(BaseEstimator,TransformerMixin):
    def __init__(self):
        return None
    def fit(self,x):
        return self
    def transform(self,x):
        mu = np.mean(x)
        sigma = np.std(x)
        z = (x - mu)/sigma
        return z
