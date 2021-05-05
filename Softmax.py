import numpy as np

class SoftmaxClassifier:
    """
    Softmax regression (classifier) used for multiclass classification
    used softmax loss.

    Input:
    X:(array[N, D]) N smaple, D feature(dimention) add bias as [N, D+1]
    y:(array[C+1, 1]) value as (0, 1, 2, ... , C) so total C+1 class
        after OneHot->Y(arragy[C+1, N])=[y1, y2 ... yN]
    iteration: int

    Params:
    Theta:(array[D+1, C+1])

    Output:
    h:(array[C+1, N])
    """
    def __init__(self, X, y, alpha):
        self.N, D = X.shape
        self.X = np.c_[X, np.ones(self.N)]
        self.C = np.max(y)
        self.Y = self.OneHot(y)
        self.alpha = alpha
        self.Theta = np.random.randn(D+1, self.C+1)

    def OneHot(self, y):
        onehot_y = np.empty([self.C+1, self.N])
        for i in range(self.N):
            onehot_y[y[i], i] = 1
        return onehot_y

    def forward(self, iteration):
        for it in range(iteration):
            # z: [C+1, N]=[N, D+1].dot([D+1, C+1])
            z = (self.X.dot(self.Theta)).transpose()
            # sum z: [1, N]; h: [C+1, N]/[1, N]=[C+1, N]
            self.h = z/np.sum(z, axis=0)
            self.Optim()
            print(f'======>Iteration:{it}')

        return self.h

    def Optim(self):
        # [C+1, N].dot([N, D+1]) = [C+1, D+1]
        grad = (self.h - self.Y).dot(self.X)
        self.Theta -= self.alpha * grad.transpose()

    def Loss(self):
        pass

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

if __name__=='__main__':
    samples = 50
    features = 6
    alpha = 0.1
    X, y = make_classification(samples, features, n_informative=3, n_classes=3)
    num_it = 100
    model = SoftmaxClassifier(X, y, alpha)
    h = model.forward(num_it)

    plt.style.use('seaborn')
    plt.scatter(X[0], X[1])
    plt.show()
