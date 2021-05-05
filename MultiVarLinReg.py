import numpy as np

class MulVarLinReg:
    """
    Input:
    X:  (list[N, M]) X=[x_1, x_2, ...x_M]; x_i=list[N];
        N: smaples; M: features.

    Params:
    Theta:(list[1, M]) Theta=[theta_1, ...]
    b: float

    Output:
    h:  (list[N, 1]) 
        h = X * Theta^T=theta_1^T*x_1 + the_2^T*x_2 + ... + theta_M^T*x_M

    """
    def __init__(self, X, y, alpha):
        N, M = X.shape
        self.X = X
        self.y = y
        self.alpha = alpha
        self.Theta = np.random.randn(1, M)
        #self.b = np.random.randn(1)
        self.b = 0.0

    def forward(self, iteration):
        for it in range(iteration):
            l = 0
            self.h = self.X.dot(self.Theta.transpose())+self.b
            self.Opt_Grad()
            l = self.Loss()
            print(f'======>Iter:{it}, loss:{l:.8f}')
            if l < 1e-8:
                break

        return self.h, self.Theta, self.b

    def predict(self, sample_test):
        # sample_test: list[1, M]
        y_pred = sample_test.dot(self.Theta.transpose())+self.b
        return y_pred

    def Loss(self):
        J = 0.5*np.sum((self.h-self.y)**2)/len(self.y)
        return J

    def Opt_Grad(self):
        # grad: list[1,M]
        grad = list()
        grad = (self.h-self.y).transpose().dot(self.X)
        self.Theta -= grad * self.alpha

    def Opt_Norm(self):
        pass


import matplotlib.pyplot as plt

if __name__=='__main__':
    samples = 3
    features = 6
    X = np.random.rand(samples, features)
    y = np.random.rand(samples, 1)

    # train
    alpha = 0.05
    iteration = 10000
    model = MulVarLinReg(X, y, alpha)
    h, theta, bias = model.forward(iteration)
    
    # test
    sample_test = np.random.rand(1, features)
    y_pred = model.predict(sample_test)
    #print(f'predict y is {y_pred:.6f}')
    print(y_pred)

    # plot
    lin = [i for i in range(X.shape[0])]
    plt.scatter(lin, y)
    plt.scatter(lin, h)
    #lar = np.arange(X.shape[0])
    #plt.bar(lar+0.23, y.flatten(), width=0.5)
    #plt.bar(lar-0.23, h.flatten(), width=0.5)
    
    plt.tight_layout()
    plt.show()
