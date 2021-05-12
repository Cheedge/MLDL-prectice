import numpy as np

class GaussianDiscrimativeAnalysis_2classes:
    """
    using X, y => construct X_hat => cal Pybyx

    Input:
    X: (array [N, D])
    y: (array [N,])

    Params:
    phi: (array [1, 1]), phi=(1/N)*sum_1->N y_i 1{y_i==1}
    mu: (array [1, D]), mu=sum_1->N X_i 1{y==k}/ sum__1->N 1{y_i==k}
    sigma: (array [D, D]), sigma=(1/N) sum_1->N (xi-mu_k)^T(xi-mu_k) 1{y--k}

    Variable:
    Px:
    Pxy:
    Py:

    Output:
    P: 
    """
    def __init__(self, X, y):
        N, D = X.shape
        # sum yi 1{yi==1}
        sumy = np.sum(y)
        # sum 1{yi==1}
        sum1 = np.sum(y)
        # sum xi 1{yi==1}
        x0 = np.array([X[i] for i in range(N) if y[i]==0])
        x1 = np.array([X[i] for i in range(N) if y[i]==1])
        sumx0 = np.sum(x0, axis=0)
        sumx1 = np.sum(x1, axis=0)
        #sumx1 = np.sum([X[i] if y[i]==1 else np.zeros_like(X[0]) for i in range(N)], axis=0)
        #sumx0 = np.sum([X[i] if y[i]==0 else np.zeros_like(X[0]) for i in range(N)], axis=0)
        self.phi = (1/N) * sumy
        self.mu0 = sumx0/(N-sum1)
        self.mu1 = sumx1/sum1

        # sum (xi-mu0) 1{yi==0}
        sumx_mu0 = (x0-self.mu0).transpose().dot((x0-self.mu0))
        # sum (xi-mu1) 1{yi==1}
        sumx_mu1 = (x1-self.mu1).transpose().dot((x1-self.mu1))

        self.sigma = (1/N) * np.sum((sumx_mu0 + sumx_mu1),keepdims=True)

    """
    P(x|y=0)=N(mu0, sigma)
    P(x|y=1)=N(mu1, sigma)
    P(y)=Ber(0, 1)=phi
    """
    
    #def cal_Pxbyy(self):
    #    pass

    #def cal_Py(self):
    #    self.py = self.phi

    #def cal_Pybyx(self):
    #    pass

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

if __name__=='__main__':

    pass
