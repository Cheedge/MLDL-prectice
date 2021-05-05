import numpy as np

class KMeans():
    """
    initial Input:
    K:(int) num of clusters

    forward Input:
    x: dataset(list[N, dim]) data
    forward Output:
    mu:(list[K, dim]) centroid of K clusters

    Process:
    1, init K cluster centroid mu[0:K]
    2, Repeat{
            loop over all points N, find N-points centroid index c_idx[0:N]. c_idx[i]=minarg ||x[i]-mu[ki]||^2
            loop over all K clusters find average(mean) of points assigned to clusters. mu[ki]=1/m sum x[i]
            }
    """

    def __init__(self, K, dim):
        # intialize K cluster centroid
        self.mu = np.random.rand(K, dim)
        self.K = K

    """
    cal optim func
    optim Input:
    x, mu, c_idx
    optim Output:
    J: cost (distortion) func(list[K])
    average distance of every point and its corresponding cluster point mu
    (sum ||x[c_idx] - mu[:, dim]||**2) / N
    """
    def optim(self, x_after_cluster, c_idx, mu):
        pass


    def forward(self, x):
        N = len(x)
        iterations = 1000
        cost = 0.0
        # repeat loops
        for it in range(iterations):
            J = 0
            distance = np.empty(self.K)
            c_idx = np.empty(N)
            # loop x
            for i in range(N):
                distance = (x[i][0] - self.mu[:, 0])**2+(x[i][1] - self.mu[:, 1])**2
                c_idx[i] = int(np.argmin(distance))

            # loop K
            for ki in range(self.K):
                x_after_cluster = list()
                # x_after_cluster: m * [2]
                x_after_cluster = [x[i] for i in range(N) if int(c_idx[i]) == ki]
                if len(x_after_cluster)==0:
                    continue
                self.mu[ki] = np.sum(x_after_cluster, axis=0)/len(x_after_cluster)
                # optim
                J += np.sum((x_after_cluster[:] - self.mu[ki])**2)
            J /= N
            print(f'================>Num K:{self.K}, Iteration:{it}, cost:{J:.16f}')
            if np.abs(cost - J)<1e-16:
                break
            else:
                cost = J

        return self.mu, cost

import matplotlib.pyplot as plt
if __name__ == '__main__':
    x = np.random.rand(100, 2)
    K, init_num = 9, 10
    cost_best, mu_best = list(), list()

    for ki in range(2, K):
        # try more initial Kpoint
        cost_min = 100
        for ii in range(init_num):
            method = KMeans(ki, 2)
            mu, cost = method.forward(x)
            if cost < cost_min:
                cost_min = cost
                mu_min = mu
        cost_best.append(cost_min)
        mu_best.append(mu_min)

    best_k_idx = np.argmin(cost_best)
    best_mu = mu_best[best_k_idx]
    
    plt.scatter(x[:, 0], x[:, 1])
    plt.scatter(best_mu[:, 0], best_mu[:, 1])
    plt.show()
