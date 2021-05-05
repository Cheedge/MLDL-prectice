import numpy as np

class MulClasLogReg:
    """
    Input:
    X:(list[N,M])
    y:(list[N])

    Params:
    self.X: (list[N, M+1])
    Theta: (list[1, M+1])
    expo: self.X*Theta^T: ([N, 1]) 
    h: 1/(1+exp(-Theta*X)) ([N, 1])
    """
    def __init__(self, X, y, alpha):
        N, M = X.shape
        self.X = np.c_[X, np.ones(N)]
        self.y = np.array(y).reshape(N, 1)
        self.Theta = np.random.randn(1, M+1)
        self.alpha = alpha


    def forward(self, iteration):
        for it in range(iteration):
            # [N, 1] = [N, M+1]dot[M+1, 1]
            s = self.X.dot(self.Theta.transpose())
            self.h = 1.0/(1+np.exp(-s))
            self.Optim()
            l = self.Loss()
            print(f'======>Iter:{it}; loss:{l:.6f}')
            if np.abs(l)<1e-6:
                break
        return self.h, self.Theta

    def Loss(self):
        # [N, 1]^T*[N, 1]
        J = (1-self.y).transpose().dot(np.log(1-self.h)) + self.y.transpose().dot(np.log(self.h))
        return float(J)

    def Optim(self):
        # error: [N,1]
        error = self.h - self.y
        # grad = [1, M+1]=[N, 1]^T*[N, M+1]
        grad = error.transpose().dot(self.X)
        # [1, M+1]-= [1, M+1]*1
        self.Theta -= grad * self.alpha

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

if __name__=='__main__':
    samples = 100
    features = 6
#    X = np.random.rand(samples, features)
#    y = [(1 if np.random.randn(1)>0 else 0) for i in range(samples)]
    X, y = make_classification(samples, features)
    iteration = 1000
    alpha = 0.1
    model = MulClasLogReg(X, y, alpha)
    h, theta = model.forward(iteration)

    plt.scatter(X[:,0], X[:, 1])
    #plt.scatter(h, y)
    plt.tight_layout()
    plt.show()
