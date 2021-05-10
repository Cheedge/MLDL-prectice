import numpy as np
"""
ref:http://krasserm.github.io/2019/02/23/bayesian-linear-regression/
"""

class BayesianLinearRegression:
    """
    BayesianLR: produce theta distirbution and use this disribution can do:
        1, make samples of theta for simulation
        2, from theta distri -> y predict distribution N(y_pred_mean, y_pred_var)
    Input:
    X: (array, [N, D])
    y: (array, [N,])
    sigma: float
    tau: float
    x_line: (array, [M, 1])

    Params:
    A, tau2I_inv: [D, D]
    X_bias = ([N, D+1]): bias means y=w1*x+b, here b=1
    x_line_bias: ([M, 2])
    
    Output:
    Theta_mu: ([D, 1])
    Theta_cov: ([D, D])
    y_pred_mean: ([M, 1])
    y_pred_var: ([M, 1])
    """
    def __init__(self, X, y, sigma, tau):
        N, D = X.shape
        X_bias = np.c_[X, np.ones(N)]
        self.Phi = X_bias
        self.y = y
        self.sigma = sigma
        self.tau = tau
        tau2I_inv = (1/tau**2) * np.eye(D+1) #np.diag(np.ones(D))
        self.A = X_bias.transpose().dot(X_bias)/sigma**2 + tau2I_inv
        # [D+1, D+1]
        self.Ainv = np.linalg.inv(self.A)
        # [D+1, D+1]@[D+1, N]@[N, 1] = [D+1, 1]
        self.AinverXTy = self.Ainv @ X_bias.transpose() @ y /sigma**2

    def Posterior(self):
        Theta_mu = self.AinverXTy
        Theta_cov = self.Ainv
        #self.PthetaS = 1/(2*np.pi*(self.Ainv)**(0.5)) * np.exp(-0.5 * (np.random.rand(D)-self.AinverXTy)**2/self.Ainv)
        return Theta_mu, Theta_cov

    def predict(self, x_line):
        M = x_line.shape[0]
        x_line_bias = np.c_[x_line, np.ones(M)]
        # [M, D+1]@[D+1, 1] = [M, 1]
        #print(M, x_line_bias.shape,self.AinverXTy.shape)
        y_pred_mean = x_line_bias @ self.AinverXTy
        # [M, D+1]@[D+1, D+1]*[M, D+1] = [M, D+1]
        # sum->[M, 1]
        y_pred_var = np.sum(x_line_bias @ self.Ainv * x_line_bias, axis=1).reshape(-1, 1) + self.sigma**2
        return y_pred_mean, y_pred_var

    def Log_Marginal_Likelihood(self):
        N, num_basis = self.Phi.shape
        Theta_mu = self.AinverXTy
        Theta_cov = self.Ainv
        # notice log(A) should use log(det(A))
        A = np.linalg.det(self.A)
        sigma, tau = self.sigma, self.tau

        Edw = np.sum((self.y - self.Phi @ Theta_mu)**2)/(2*sigma) + np.sum(Theta_mu.transpose() @ Theta_mu)/(2*tau)
        score = -N * np.log(self.sigma) - num_basis * np.log(self.tau) - 0.5 * np.log(A) - 0.5 * N * np.log(2 * np.pi) - Edw
        return score
