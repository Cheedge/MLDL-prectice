import numpy as np
"""
Cal Independent Component Analysis
1,  P_x(x)=Π P_s(Wj^T X)|W|
    choose P_s: logistic distri, laplace distri ...
2,  log likelihood
    l(W) = Σ_i(Σ_j log P_s(Wj^T X) + log|W|)
    ∇l(W) = 0 ⇒ W
3, Stochastic gradient ascent
    W := W + α()

Learn from:
    CS229 ICA
    and
    https://github.com/akcarsten/Independent_Component_Analysis
"""

class ICA:
    """
    X_j = Σ_i A_ji S_i
    S = W X

    Input:
    X: ([N, T]) N mico, T time

    Output:
    W: ([T, N, N])
    S: ([N, T]) N speaker, T time
    """
    def __init__(self, X, alpha):
        N, T = X.shape
        self.X = X
        self.W = np.random.rand(T, N, N)
        self.alpha = alpha
        self.S = np.zeros([N, T])

    @staticmethod
    def Logistic_Distri(x):
        cdf = 1.0/np.exp(-x)
        pdf = cdf * (1.0 - cdf)
        return cdf, pdf

    @classmethod
    def Px(cls):
        _, pdf = Logistic_Distri()
        pass

    @classmethod
    def LogL(cls, w):
        return grad_l
        pass

    @staticmethod
    def StochGrad(w, alpha):
        return w
        pass

    @classmethod
    def forward(cls):
        X, W = self.X, self.W
        while True:
            Px()
            LogL()
            StochGrad()
            if np.abs(W-W0) < eps:
            break
        return S
