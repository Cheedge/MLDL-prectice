import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
"""
plot the Bayesian confidence region, posterior fitting and predictions.
here use one-dim X, and y, so X should be preprocessed by dimenitonality reduction method as PCA, 
and the prediction y = W1*X+w0*1
X: original data, without bias b=1
X_bias: add bias data, np.c_[X, np.ones(X.shape)]
x_line_test: continuous data without bias, used for plot predict y_hat

ref:http://krasserm.github.io/2019/02/23/bayesian-linear-regression/
"""

def PlotBayesianConfidenceRegion(x_line_test, y_mean, y_std):
    """
    x_line_test: ([M, D]=[M, 1]), without bias
    y_mean: ([M, 1])
    y_std: ([M, D]=[M, 1])
    """
    #polygon_class = {alpha: 0.5, label:"confidence"}
    plt.plot(x_line_test, y_mean, '-')
    #plt.fill_between(x_test, y_mean - y_std, y_mean + y_std, polygon_class)
    y1 = y_mean - y_std
    y2 = y_mean + y_std
    plt.fill_between(x_line_test.ravel(), y1.ravel(), y2.ravel(), alpha=0.5)

def PlotPosteriorDistribution(Theta_mu, Theta_cov, num_grids):
    """
    Theta_mu: [D, 1]
    Theta_cov: [D, D]
    """
    w0_min, w0_max, w1_min, w1_max = -3, 3, -3, 3
    # [num,]
    w0 = np.linspace(w0_min, w0_max, num_grids)
    w1 = np.linspace(w1_min, w1_max, num_grids)
    # meshgrid: [num, num]; dstack: [num, num, 2]
    # reshape: [num*num, 2]
    theta_grids = np.dstack(np.meshgrid(w0, w1)).reshape(-1,2)

    # multivarizt_normal([1, D],[D,D])
    # Array 'mean' must be a vector of length 2.
    norm_distri = stats.multivariate_normal(Theta_mu.ravel(), Theta_cov)
    density = norm_distri.pdf(theta_grids).reshape(num_grids, num_grids)
    plt.imshow(density, origin='lower', extent=(w0_min, w0_max, w1_min, w1_max))



def PlotPosteriorFitting(x_line, Theta_mu, Theta_cov, num_theta, axis_x_line=None):
    """
    x_line: ([N, d] = [N, 1])after dimentionality reduction
    x_line_bias: ([N, d+1]=[N, 2]) with bias=1
    Theta_mu: ([D, 1]=[2, 1])
    Theta_cov: ([D, D]=[2, 2])
    """
    # here d=1, D=2
    # multivariate_normal: mean must be 1 dimensional
    #x_line = x_line.reshape(-1, 1)
    D = Theta_mu.shape[0]
    #d = x_line.shape[1]
    m = num_theta
    # [N, d] -> [N, d+1]
    x_line_bias = np.c_[x_line, np.ones(x_line.shape[0])]
    #print(x_line_bias.shape)
    thetas = np.random.multivariate_normal(Theta_mu.ravel(), Theta_cov, size=m)
    for i in range(m):
        # [m, D] -abstract> [1, D]
        theta_i = thetas[i].reshape(-1,D)
        #print(x_line_bias.shape, thetas.shape, thetas[i].shape)
        # [N, d+1]@[D, 1]=[N, 1]
        y_i = x_line_bias @ theta_i.transpose() 
        #print(x_line_bias.shape, y_i.shape, theta_i.shape)
        if axis_x_line is None:
            axis_x_line = x_line
        plt.plot(axis_x_line.ravel(), y_i.ravel(), 'r-')
