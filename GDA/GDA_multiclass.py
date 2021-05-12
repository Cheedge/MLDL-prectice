import numpy as np
from collections import Counter

class GaussianDiscrimativeAnalysis_Multiclasses:
    """
    using X, y => construct X_mu, X_var => cal Pybyx

    Input:
    X: (array [N, D])
    y: (array [N,]) because multiclasses so value [0,...,C]

    Output 1:
    cal these params to make X distribution for prediction.
    
    phi: (list [C]), phi=(1/N)*sum_1->N y_i 1{y_i==1}
    mu: (array [C, D]), mu=sum_1->N X_i 1{y==k}/ sum__1->N 1{y_i==k}
        mu_k: [1, D]
    var: (array [C, D, D]), var=(1/N) sum_1->N (xi-mu_k)^T(xi-mu_k) 1{y==k}
        var_k: [1, D, D]

    Variable:
    Pxbgy: P(x|y) [C, n]
           P(x|y=k) [n,]
    Py: P(y) phi
    Pybgx: P(y|x) [C, 1]

    Output:
    y: int value[0, C) 
    """
    def __init__(self, X, y):
        """
        P(x|y=k)=N(mu_k, var_k)
        P(y)=Ber(0, ... , C-1) = phi_k
        """
        self.X = X
        self.y = y
        N, D = X.shape
        dic = Counter(y)
        self.C = len(dic.keys())
        self.phi = list()
        self.mu = np.zeros((self.C, D))
        self.var = np.zeros((self.C, D, D))
    
    def SeparateXpoints(self):
        N, _ = self.X.shape
        dic = dict()
        for k in range(self.C):
            xk = np.array([self.X[i] for i in range(N) if self.y[i]==k])
            dic.update({k: xk})
        return dic


    def GenerativeModel(self):
        N, D = self.X.shape
        sm = np.zeros((D, D))
        for k in range(self.C):
            # [n, D]
            x_k = np.array([self.X[i] for i in range(N) if self.y[i]==k])
            # sum 1{yi==k}: n
            sum1_k = x_k.shape[0]
            self.phi.append(sum1_k / N)
            # sum xi 1{yi==k}: [1, D]
            sumx_k = np.sum(x_k, axis=0)
            self.mu[k] = sumx_k / sum1_k
            # sum (xi-mu1) 1{yi==1}: [n, D] - [1, D] = [n, D]
            x_minus_mu = x_k - self.mu[k]
            #sm = np.sum(x_minus_mu.transpose() @ x_minus_mu, axis=0)
            sm = x_minus_mu.transpose() @ x_minus_mu
            # [D, D]
            self.var[k] = sm / sum1_k 

        return self.mu, self.var, self.phi

           
    def prediction(self, x_hat):
        """
        P(x_hat|y=k)
        formular see https://scikit-learn.org/stable/modules/lda_qda.html

        P(y_pred|x_hat)=P(x_hat|y_pred)P(y_pred)/P(x_hat)=>
        argmax(P(y_pred|x_hat)) = argmax(sum P(x_hat|y_pred=k)P(y_pred=k))
        """
        M, D = x_hat.shape
        Pybgx = np.zeros((self.C, M))
        for k in range(self.C):
            coeff = 1/(np.sqrt(2*np.pi)**D * np.sqrt(np.linalg.det(self.var[k])))
            var_inv = np.linalg.inv(self.var[k])
            # [M, D] - [1, D] = [M, D]
            xhat_minus_mu = x_hat - self.mu[k]
            # XT @ S_inv @ X = sum(X @ S_inv * X, axis=1): [M, D]->[M,]
            exppart = np.exp(-0.5 * np.sum(xhat_minus_mu @ var_inv * xhat_minus_mu, axis=1)).ravel()
            # P(x_hat|y_pred=k): [M,]
            Pxbgy_k = coeff * exppart 
            # P(y_pred)=phi_k
            # P(y_pred|x_hat) = P(x_hat|y_pred=k)*P(y_pred)
            # [M,]
            Pybgx[k] = Pxbgy_k * self.phi[k]
    

        # argmax([C, M], axis=0): [1, M]
        y_pred = np.argmax(Pybgx, axis=0)
        return y_pred

#import matplotlib.pyplot as plt
#from sklearn.datasets import load_iris
#from sklearn.model_selection import train_test_split
#from sklearn.decomposition import PCA
#from sklearn.metrics import accuracy_score, classification_report
#
#if __name__=='__main__':
#
#    pass
