import numpy as np
from collections import Counter

class GaussianDiscrimativeAnalysis_Multiclasses:
    """
    using X, y => construct X_hat => cal Pybyx

    Input:
    X: (array [N, D])
    y: (array [N,]) because multiclasses so value [0,...,C]

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
        dic = Counter(y)
        C = len(dic.keys())
        sm = np.zeros(D, D)
        for k in range(C):
            # [n, D]
            x_row[k] = np.array([X[i] for i in range(N) if y[i]==k])
            # sum 1{yi==k}
            sum1[k] = x_row[k].shape[0]
            # [1, D] sum xi 1{yi==k}
            sumx[k] = np.sum(x_row[k], axis=0)
            self.mu[k] = sumx[k] / sum1[k]
            self.phi[k] = sum1[k] / N
            # sum (xi-mu1) 1{yi==1}
            xi_minus_muk = x_row[k] - self.mu[k]
            sm += np.sum(xi_minus_muk.transpose() @ xi_minus_muk, axis=0)
        self.sigma = (1/N) * sm 
        #self.sigma = (1/N) * np.sum((sumx_mu0 + sumx_mu1),keepdims=True)

    """
    P(x|y=0)=N(mu0, sigma)
    P(x|y=1)=N(mu1, sigma)
    P(y)=Ber(0, 1)=phi
    """
    
    def cal_Pxgby(self):
        exp0 = np.exp(-0.5)
        self.pxgby = (1/(np.sqrt(2*np.pi)*self.sigma)*exp_0)
        pass

    def cal_Py(self):
        self.py = self.phi

    def cal_Pygbx(self):
        """
        P(y|x)=P(x|y)P(y)/P(x)=>
        argmax(P(y|x)) = argmax(P(x|y)P(y))
        """
        self.pygbx = self.pxgby * self.py

    def perdiction(self, x_line):
        y_pred = np.argmax(Pygbx)
        return y_pred

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

if __name__=='__main__':

    pass
