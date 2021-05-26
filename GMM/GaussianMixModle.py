import numpy as np
from scipy.stats import multivariate_normal
"""
calculate Gaussian Mixed Model
    Qi(zi==j)=P(zi==j|xi,theta)
    zi ~ Multinomial(phi_i)
    xi|zi==j ~ N(mu_j, cov_j)
"""
class GMM:
    '''
    Input:
    X:([ N, D ])
    K:(int)

    Params:
    w:([ N, K ])
    mu:([ D ]-->[ K, D ])
    cov:([ D, D ]-->[ K, D, D ])
    phi:([ N ]-->[ K, N ])

    Output:
    z:([ N ])
    '''
    def __init__(self, X, K):
        self.X = X
        self.K = K
        N, D = X.shape
        self.mu = np.random.rand(K, D)
        self.cov = np.random.rand(K, D, D)
        self.phi = np.random.randn(K, N)
        self.z_hat = np.empty(N)


    def E_step(self):
        '''
        Qi(zi==j)=P(zi==j|xi,theta)
        use Bayesian
        P(zi|xi,theta)=P(xi|zi)P(zi)

        zi ~ Multinomial(phi_i)
        xi|zi==j ~ N(mu_j, cov_j)
        '''
        N, D = self.X.shape
        self.w = np.empty(N, self.K)
        Pzjbx = np.empty(N, self.K)
        sum_Pxbzj = 0
        for j in range(1, self.K+1):
            # [ N ]
            Pzj = self.phi
            # sum([ N, D ]@[ D, D ]*[ N, D ], axis=1)=sum([ N, D ],1)=[ N ]
            #Pxbzj = 1/((2*np.pi)**(D/2)*np.sqrt(self.cov[ j ])) * np.exp(-0.5*np.sum((self.X-self.mu[ j ])@np.invert(self.cov[ j ])*(self.X - self.mu[ j ]), axis=1)).ravel()
            Pxbzj = multivariate_normal.pdf(self.X, mean=self.mu[ j ], cov=self.cov[ j ])
            # [ N ]
            Pzjbx[ :, j ] = Pxbzj * Pzj
            # [ N ]
            sum_Pxbzj += Pzjbx[ :, j ]
            self.w[ :, j ] = Pzjbx[ :, j ]/sum_Pxbzj
        return self.w, sum_Pxbzj

    def M_step(self):
        '''
        update mu, cov, phi
        phi_j = (1/N) *sum_i wj^i
        mu_j = sum_i wj^i*xi /sum_i wj^i
        cov_j = sum_i (xi-muj)T(xi-muj)* wj^i / sum_i wj^i
        '''
        N, D = self.X.shape
        sum_w = np.sum(self.w, axis=0)
        for j in range(1, self.K+1):
            self.phi[ j ] = (1/N) * sum_w[ j ]
            # wj:[ N ]; X:[ N, D ]; mu:[ D ]
            self.mu[ j ] = np.sum(self.X*self.w[ j ], axis=0)/sum_w[ j ]
            self.cov[ j ] = np.sum(np.transpose(self.X - self.mu)@(self.X -self.mu)*self.w[ j ], axis=0)/sum_w[ j ]

    def LogLikelihood(self):
        '''
        joint likelihood P(x, z)=P(x|z)P(z)
        '''
        #return lh
        pass

    def forward(self):
        lt0 = 0
        while(True):
            omega, l = E_step()
            lt = np.log(l)
            M_step()
            #lt = LogLikelihood()
            if np.abs(lt - lt0) < eps:
                break
            else:
                lt0 = lt
        # omeag[ N, K ]
        pred_z = np.argmax(omega, axis=1)
