import numpy as np

class BayesianLinearRegression:
    """
    BayesianLR: produce theta distirbution and use this disribution can do:
        1, make samples of theta for simulation
        2, from theta distri -> y predict distribution N(y_pred_mean, y_pred_std)
    Input:
    X: (array, [N, D])
    y: (array, [N,])
    sigma: float
    tau: float
    x_hat: (array, [M, D])
    
    Output:
    y_pred_mean: ([M, 1])
    y_pred_std: ([M, 1])
    """
    def __init__(self, X, y, sigma, tau):
        N, D = X.shape
        self.X = X
        self.y = y
        self.sigma = sigma
        tau2I = (tau**2) * np.diag(np.ones(D))
        A = 1/sigma * X.transpose().dot(X) + 1/ tau2I) 
        # [D, D]
        self.Ainv = np.linalg.inv(A)
        # [D, D]@[D, N]@[N, 1] = [D, 1]
        self.AinverXTy = self.Ainv @ X.transpose() @ y /sigma**2

    #def cal_PthetaS(self):
    #    self.PthetaS = 1/(2*np.pi*(self.Ainv)**(0.5)) * np.exp(-0.5 * (np.random.rand(D)-self.AinverXTy)**2/self.Ainv)

    def predict(self, x_hat):
        # [M, D]@[D, 1] = [M, 1]
        y_pred_mean = x_hat @ self.AinverXTy
        # [M, D]@[D, D]*[M, D] = [M, 1]
        y_pred_std = x_hat @ self.Ainv * x_hat + self.sigma**2
        return y_pred_mean, y_pred_std

