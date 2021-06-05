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
        #self.mu = np.random.rand(K, D)
        # [ K, D ]choose one point of every row as mean
        self.mu = np.random.choice(X.flatten(), (K, D))
        self.cov = self.makePSD(K, D)
        # [ K, N ]
        self.phi = np.repeat((np.ones(K)/K).reshape(K, 1), N, axis=1)
        self.z_hat = np.zeros(N)
        self.eps = 1E-6

    @staticmethod
    def makePSD(K, D):
        a = np.random.randn(D, D)
        s_matrix = a@a.T
        s_matrix = s_matrix[np.newaxis, :]
        for i in range(K-1):
            A = np.random.rand(D, D)
            # A.T@A or A@A.T is PSD
            s = A@A.T
            s_matrix = np.concatenate((s_matrix, s[np.newaxis, :]), axis=0)
        return s_matrix

    def E_step(self):
        '''
        Qi(zi==j)=P(zi==j|xi,theta)
        use Bayesian
        P(zi|xi,theta)=P(xi|zi)P(zi)

        zi ~ Multinomial(phi_i)
        xi|zi==j ~ N(mu_j, cov_j)

        '''
        N, D = self.X.shape
        self.w = np.zeros([N, self.K])
        Pzjbx = np.zeros([N, self.K])
        reg_cov = self.eps * np.identity(D)
        for j in range(self.K):
            sum_j_Pxbzj = 0
            # [ N ]
            Pzj = self.phi[j]
            # sum([ N, D ]@[ D, D ]*[ N, D ], axis=1)=sum([ N, D ],1)=[ N ]
            # notice: np.invert: element invert.
            #coeff = 1/((2*np.pi)**(D/2)*np.abs(np.linalg.det(self.cov[ j ])))
            #exppart = np.exp(-0.5*np.sum((self.X-self.mu[ j ])@np.linalg.inv(self.cov[ j ]+reg_cov)*(self.X - self.mu[ j ]), axis=1))
            #Pxbzj =  coeff * exppart.ravel()
            # must be positive semidefined(PSD) matrix: means invertable
            # [ N ]
            Pxbzj = multivariate_normal.pdf(self.X, mean=self.mu[ j ], cov=self.cov[ j ]+reg_cov)
            # [ N ]
            Pzjbx[ :, j ] = Pxbzj * Pzj
            # [ N ]
            #sum_j_Pxbzj += Pzjbx[ :, j ]
            #self.w[ :, j ] = Pzjbx[ :, j ]/(sum_j_Pxbzj+self.eps)
        # sum over i and j: float
        sum_Pxbzj = np.sum(Pzjbx)
        # [N, K]/float = [N, K]
        self.w = Pzjbx/sum_Pxbzj
        #print(sum_Pxbzj, Pzjbx.shape)
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
        if rank(X|z)<N<D:
        Singularity!
        '''
        N, D = self.X.shape
        # sum[ N, K ]->[ K ]
        sum_w = np.sum(self.w, axis=0)
        for j in range(self.K):
            self.phi[ j ] = (1/N) * sum_w[ j ]
            # wj:[ N ]; X:[ N, D ]; mu:[ D ]
            # notice: should be [N, D]*[N, 1] not [N, D]*[N,]
            self.mu[ j ] = np.sum(self.X*self.w[ :, j ].reshape(N, 1), axis=0)/(sum_w[ j ]+self.eps)

            # [N, D]T[N, D]
            #self.cov[ j ] = (self.X - self.mu[j]).transpose()@((self.X -self.mu[j])*self.w[ :, j ].reshape(N, 1))/(sum_w[ j ]+self.eps)
            covj = np.zeros([D, D])
            for i in range(N):
                # X_i:[D], mu_j:[D], w^i_j:[N, K]->1
                covj += (self.X[i] - self.mu[j]).reshape(D, 1)@((self.X[i] -self.mu[j]).reshape(1, D))*self.w[i, j]/(sum_w[ j ]+self.eps)
            self.cov[j] = covj

    def LogLikelihood(self, z):
        '''
        Joint likelihood: sum_i P(xi, zi)=P(xi|zi)P(zi)
        zi=j->[0,K)
        P(zi) = phi(zi)
        P(xi|zi) = N(mu_zi, cov_zi)

        '''
        N, D = self.X.shape
        # [K, D]->[N, D]
        m = self.mu[z]
        # [K, D, D]->[N, D, D]
        s = self.cov[z]
        '''
        coeff_l = 1/((2*np.pi)**(D/2)*np.sqrt(np.abs(np.linalg.det(s))))
        # sum[N, D]@[N, D, D]*[N, D]->[N]
        exppart_l = np.exp(-0.5*np.sum((self.X-m)@np.linalg.inv(s)*(self.X - m), axis=2))
        print(exppart_l.shape, '\n',coeff_l.shape, '\n')
        Pxbgz = coeff_l * exppart_l
        # [N]
        print(Pxbgz.shape, '\n',Pzi.shape)
        Pxz = Pxbgz * Pzi
        # float
        lh = np.sum(Pxz, axis=0)
        '''
        lh = 0
        for i in range(N):
            # [K, N]->[N]
            Pzi = self.phi[z[i], i]
            #coeff_l = 1/((2*np.pi)**(D/2)*np.abs(np.linalg.det(s[ i ])))
            ## sum[D]@[D, D]*[D]->float
            #exppart_l = np.exp(-0.5*np.sum((self.X[ i ]-m[ i ])@np.linalg.inv(s[ i ]+self.eps*np.identity(D))*(self.X[ i ] - m[ i ]), axis=0))
            #Pxbgz = coeff_l * exppart_l
            #Pxz = Pxbgz * Pzi[i]

            Pxbgz = multivariate_normal.pdf(self.X[ i ], m[ i ], s[ i ]+self.eps*np.identity(D))
            Pxz = Pxbgz * Pzi

            lh += Pxz
        loglh = np.log(lh)
        return loglh

    def forward(self):
        lglt0 = 0
        eps = 1E-6
        while(True):
            omegas = self.E_step()
            self.M_step()

            # omeags[ N, K ]
            pred_z = np.argmax(omegas, axis=1)
            loglt = self.LogLikelihood(pred_z)
            print('in code', pred_z)
            if np.abs(loglt - lglt0) < eps:
                break
            else:
                lglt0 = loglt
        return pred_z
