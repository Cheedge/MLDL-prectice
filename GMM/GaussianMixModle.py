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
        self.mu = np.random.randn(K, D)
        self.cov = np.random.randn(K, D, D)
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
        self.w = np.empty([N, self.K])
        Pzjbx = np.empty([N, self.K])
        for j in range(self.K):
            sum_j_Pxbzj = 0
            # [ N ]
            Pzj = self.phi[j]
            # sum([ N, D ]@[ D, D ]*[ N, D ], axis=1)=sum([ N, D ],1)=[ N ]
            # notice: np.invert: element invert.
            coeff = 1/((2*np.pi)**(D/2)*np.sqrt(np.abs(np.linalg.det(self.cov[ j ]))))
            exppart = np.exp(-0.5*np.sum((self.X-self.mu[ j ])@np.linalg.inv(self.cov[ j ])*(self.X - self.mu[ j ]), axis=1))
            Pxbzj =  coeff * exppart.ravel()
            # must be positive semidefined(PSD) matrix.
            #Pxbzj = multivariate_normal.pdf(self.X, mean=self.mu[ j ], cov=self.cov[ j ])
            # [ N ]
            Pzjbx[ :, j ] = Pxbzj * Pzj
            # [ N ]
            sum_j_Pxbzj += Pzjbx[ :, j ]
            self.w[ :, j ] = Pzjbx[ :, j ]/sum_j_Pxbzj
        #self.w = Pzjbx/sum_Pxbzj.ravel()
        return self.w

    def M_step(self):
        '''
        update mu, cov, phi
        phi_j = (1/N) *sum_i wj^i
        mu_j = sum_i wj^i*xi /sum_i wj^i
        cov_j = sum_i (xi-muj)(xi-muj)T* wj^i / sum_i wj^i

        !!!NOTICE!!!
        here shows the "Singular cov matrix", if use matrix multiply
        as:
        so if use matrix multiplication:
        [N,D]T@[N,D]=[D,D], then sum_i => [D]
        if rank(X|z)<D<N:
        Singularity!
        '''
        N, D = self.X.shape
        sum_w = np.sum(self.w, axis=0)
        for j in range(self.K):
            covj = np.zeros([D, D])
            self.phi[ j ] = (1/N) * sum_w[ j ]
            # wj:[ N ]; X:[ N, D ]; mu:[ D ]
            # notice: should be [N, D]*[N, 1] not [N, D]*[N,]
            self.mu[ j ] = np.sum(self.X*self.w[ :, j ].reshape(N, 1), axis=0)/sum_w[ j ]
            # [N, D]T[N, D]
            #self.cov[ j ] = np.sum((self.X - self.mu[j]).transpose()@((self.X -self.mu[j])*self.w[ :, j ].reshape(N, 1)), axis=0)/sum_w[ j ]
            for i in range(N):
                print(covj,'\t')
                print(self.X[i],'\t',self.mu[j],'\n')
                # X_i:[D], mu_j:[D], w^i_j:[N, K]->1
                covj += (self.X[i] - self.mu[j]).reshape(D, 1)@((self.X[i] -self.mu[j]).reshape(1, D))*self.w[i, j]
            self.cov[j] = covj
            print(self.cov[j])

    def LogLikelihood(self, z):
        '''
        Joint likelihood: sum_i P(xi, zi)=P(xi|zi)P(zi)
        zi=j->[0,K)
        P(zi) = phi(zi)
        P(xi|zi) = N(mu_zi, cov_zi)

        '''
        N, D = self.X.shape
        # [N]
        Pzi = self.phi[z]
        # [D]
        m = self.mu[z]
        # [D, D]
        s = self.cov[z]
        coeff_l = 1/((2*np.pi)**(D/2)*np.sqrt(np.abs(np.linalg.det(s))))
        # sum[N, D]@[D, D]*[N, D]->[N]
        exppart_l = np.exp(-0.5*np.sum((self.X-m)@np.linalg.inv(s)*(self.X - m), axis=1))
        Pxbgz = coeff_l * exppart_l
        # [N]
        Pxz = Pxbgzb * Pzi
        # float
        lh = np.sum(Pxz, axis=0)
        lglh = 0
        for i in range(N):
            lglh += np.log(Pxz[i])
        return lh, lglh

    def forward(self):
        lt0 = 0
        eps = 1E-6
        while(True):
            omegas = self.E_step()
            self.M_step()

            # omeags[ N, K ]
            pred_z = np.argmax(omegas, axis=1)
            lt, loglt = self.LogLikelihood(pred_z)
            if np.abs(loglt - lt0) < eps:
                break
            else:
                lt0 = loglt
        return pred_z
