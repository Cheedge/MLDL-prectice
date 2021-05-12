import numpy as np
from matplotlib.pyplot as plt

def PlotGDA_2D(x_pca, x_pca_dic, y_pred, resolution):
    """
    use PCA decomposition reduce the dimension to 2D
    then can use this func to plot
    
    Input:
    x_pca: 2-dim data X (x0, x1) [N, 2]
    x_pca_dic: which used for plot separated class points
        by using SeparateXpoints()
    y_pred: [N, 1]
        by using prediction(x_hat)
    resolution: int

    Output:

    """

    # prepare separate x ponits for ploting
    #x_pca_dic = GDA_pca.SeparateXpoints()
    C = len(x_pca_dic)
    
    x0_min, x0_max = np.min(x_pca[:, 0]), np.min(x_pca[:,0])
    x1_min, x1_max = np.min(x_pca[:, 1]), np.min(x_pca[:,1])
    x0 = np.linspace(x0_min, x0_max, resolution)
    x1 = np.linspace(x1_min, x1_max, resolution)


    x0_grid, x1_grid = np.meshgrid(x0, x1)
    x_hat = np.c_[x0_grid.ravel(), x1_grid.ravel()]
    #y_pred = GDA_pca.prediction(x_hat)
    z = y_pred.reshape(resolution, resolution)
    for i in range(C):
	plt.scatter(x_pca_dic[i][:, 0], x_pca_dic[i][:, 1])
    plt.contourf(x0_grid, x1_grid, z, alpha=0.2)
