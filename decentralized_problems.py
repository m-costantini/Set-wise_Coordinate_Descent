import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from scipy.optimize import minimize, fmin_tnc
import math


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class QuadraticProblem():

    def __init__(self, n, dim, A, edge_to_nodes, iid, degree, c_choice):

        super(QuadraticProblem, self).__init__()
        self.n = n
        self.dim = dim
        self.A = A
        self.edge_to_nodes = edge_to_nodes
        self.E = np.shape(A)[0]
        # Generate the data
        self.P_vec = [None] * n
        self.Q_vec = [None] * n
        self.R_vec = [None] * n

        self.P_inv = [None] * n
        self.PP_inv = [None] * n
        self.P_inv_sq = [None] * n

        # Give all nodes the same distribution, just change random matrix aux
        for i in range(n):
            # Create quadratic matrix
            aux_0 = np.random.uniform(low=-0.01, high=0.01, size=(dim,dim))
            if c_choice == "few different":
                if np.mod(i,degree) == 0:
                    c = 1e15
                else:
                    c = 10 
            elif c_choice == "increasing":
                c = i+1
            elif c_choice == "random":
                c = np.random.randint(low=1, high=50)
            elif c_choice == "max":
                aux_0 = np.zeros((dim,dim))
                if i == 0:
                    c = 1e15
                else:
                    c = 10
            else:
                raise Exception("c_choice unknown")
            P = aux_0 @ aux_0.T + c * np.eye(dim) # add identity to increase eigenvalues and give more curvature
            eig_vals, _ = np.linalg.eig(P)
            if np.any(eig_vals < 0):
                raise Exception('Eigenvalues of created matrix are smaller than 0')

            # Create linear term and offset
            Q = np.zeros((dim,1))

            R = 1 # just for f(x^*) not to be 0 and have problems in the division

            self.P_vec[i] = P
            self.Q_vec[i] = Q
            self.R_vec[i] = R

            self.P_inv[i] = np.linalg.inv(P)
            self.PP_inv[i] = 0.5 * self.P_inv[i] # = np.linalg.inv(P + P.T)
            self.P_inv_sq[i] = sqrtm(self.P_inv[i])

        # All together
        Psum = sum(self.P_vec)
        Qsum = sum(self.Q_vec)
        Rsum = sum(self.R_vec)

        # Compute optimal objective value analytically
        theta_opt = np.linalg.inv(Psum + Psum.T) @ (-Qsum)
        self.analy_opt_obj_val = self.quadratic_function(theta_opt, Psum, Qsum, Rsum).flatten()

        # Compute Lipschitz constants (same procedure as for optimal stepsizes)
        self.L = [None for e in range(self.E)]
        for e in range(self.E):
            (i,j) = self.edge_to_nodes[e]
            Ql = 0.5 * (self.P_inv[i] + self.P_inv[j])
            eig_vals, _ = np.linalg.eig(Ql)
            self.L[e] = max(eig_vals)


    def quadratic_function(self, theta, P, Q, R):
        theta = np.reshape(theta,(len(theta),1))
        return theta.T @ P @ theta + Q.T @ theta + R

    def objective(self, thetas):
        obj = 0
        for i in range(len(self.P_vec)):
            obj += self.quadratic_function(thetas[i,:], self.P_vec[i], self.Q_vec[i], self.R_vec[i])
        return obj

    def Lagrangian(self, thetas, lambdas):
        obj = self.objective(thetas)
        relax = 0
        for e in range(self.E):
            (i,j) = self.edge_to_nodes[e]
            relax += lambdas[e,:].reshape((1,self.dim)) @ ( self.A[e,i]*thetas[i,:] + self.A[e,j]*thetas[j,:] ).reshape((self.dim,1))
        return obj + relax

    def arg_min_Lagran(self, i, lambdas):
        return self.PP_inv[i] @ ( - self.A[:,i].T @ lambdas - self.Q_vec[i].flatten() )

    def get_optimal_stepsizes(self):
        opt_sz = [None] * self.E
        for k in range(self.E):
            (i,j) = self.edge_to_nodes[k]
            Ql = 0.5 * (self.P_inv[i] + self.P_inv[j])
            eig_vals, _ = np.linalg.eig(Ql)
            opt_sz[k] = 1/max(eig_vals)
        return opt_sz

    def get_dual_Hessian(self, edges_to_shared_node):
        node_Hessians = [ 0.5*self.P_inv[i] for i in range(self.n) ]
        Hessian = [[None for l in range(self.E)] for m in range(self.E)] # Hessian[i,k] = Hessian[k,i]
        for l in range(self.E):
            for m in range(l+1): # l and m can take the same value 
                if (l != m) and edges_to_shared_node[l,m] > -1: # if edges l and m share a node
                    i = edges_to_shared_node[l,m]
                    Hessian[l][m] = (-1) * self.A[l,i] * self.A[m,i] * node_Hessians[i]
                    Hessian[m][l] = (-1) * self.A[l,i] * self.A[m,i] * node_Hessians[i]
                elif l == m:
                    i, j = self.edge_to_nodes[l]
                    Hessian[l][l] = (-1) * (node_Hessians[i] + node_Hessians[j])
        return Hessian



# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class LLS_Problem():
    """
    Decentralized and regularized linear least squares problem with *no* regularization:
    f_i(u_i) = (1/M) * ||X_i u_i - Y_i||_2^2 
    but with data generated as in Scaman 2017 (cosine with noise)
    """
    def __init__(self, n, dimensions, M, A, edge_to_nodes, iid):
        # Parameters
        self.n = n
        self.d = dimensions
        self.M = M
        self.A = A
        self.E = np.shape(A)[0]
        self.edge_to_nodes = edge_to_nodes
        self.iid = iid
        # Generate data    
        self.X = [None] * n
        self.Y = [None] * n
        for i in range(n):
            self.X[i] = np.random.normal(size=(self.M, self.d ))
            xi = 0.25 * np.random.normal(size=(self.M, 1))
            self.Y[i] = self.X[i] @ np.ones((self.d, 1)) + np.cos(self.X[i] @ np.ones((self.d, 1))) + xi
            if not iid: # change slope by multiplying y values
                self.Y[i] = self.Y[i] * (i+1)
        Xall = np.array(np.concatenate(self.X,axis=0))
        Yall = np.array(np.concatenate(self.Y,axis=0))
        # Analytic solution for the complete data
        theta_opt = np.linalg.inv(Xall.T @ Xall) @ (Xall.T @ Yall) # optimal theta value
        theta_opt_tile = np.tile( theta_opt.reshape((1,self.d)), (n,1) )
        self.analy_opt_obj_val = self.objective( theta_opt_tile ) # objective value for optimal theta
        # Store frequently used values 
        self.XXinv = [ np.linalg.inv( (1/self.M) * self.X[i].T @ self.X[i] ) for i in range(n) ] 
        self.XY = [ (1/self.M) * self.X[i].T @ self.Y[i] for i in range(n) ]

        # Compute Lipschitz constants (same procedure as for optimal stepsizes)
        self.L = [None for e in range(self.E)]
        for e in range(self.E):
            i, j = self.edge_to_nodes[e]
            ddfi = (-3/2) * self.XXinv[i] # the M factor is already inside XXinv
            ddfj = (-3/2) * self.XXinv[j]

            H_ee = ddfi + ddfj 

            eig_vals = abs( np.linalg.eigvals(H_ee) ) 

            self.L[e] = max(eig_vals)


    def plot_data(self,):
        if self.d == 1:
            fig, ax = plt.subplots(figsize=(6,4))
            for i, (x, y) in enumerate(zip(self.X,self.Y)):
                ax.scatter(x,y,c='C'+str(i+1))
            ax.set_xlabel(r'$x$', fontsize=12)
            ax.set_ylabel(r'$y$', fontsize=12)
            ax.set_title("Data of all nodes", fontsize=14)
            ax.grid()
        elif self.d == 2:
            print("Sorry, not implemented yet")
        else:
            print("Sorry, I can only plot data if the dimensions are 1 or 2")
        return

    def objective(self, thetas):
        MSE_per_node = [ (1/self.M) * np.linalg.norm( self.Y[i] - self.X[i] @ thetas[i,:].reshape((self.d,1)) )**2 for i in range(self.n) ]
        obj_val = sum(MSE_per_node)
        return obj_val

    def arg_min_Lagran(self, i, lambdas):
        return self.XXinv[i] @ ( self.XY[i] - 0.5 * lambdas.T @ np.expand_dims(self.A[:,i],axis=1) ) # NOTE: M factor already in self.XXinv and self.XY
    
    def Lagrangian(self, thetas, lambdas):
        # As written here, problem-independent
        # Lag_val = self.objective(thetas) + np.matrix.trace(lambdas @ self.A @ thetas) # dimension error
        Lag_val = self.objective(thetas) + np.matrix.trace(lambdas.T @ self.A @ thetas)
        return Lag_val
        
    def get_optimal_stepsizes(self,):
        # For this we compute the diagonal element of H using the formula 
        # H_ll = A_il f_i*'' + A_jl f_j*'' 
        opt_sz = []
        for e in range(self.E):
            i, j = self.edge_to_nodes[e]
            ddfi = (-3/2) * self.XXinv[i] # the M factor is already inside XXinv
            ddfj = (-3/2) * self.XXinv[j]
            
            H_ee = ddfi + ddfj 

            eig_vals = abs( np.linalg.eigvals(H_ee) )

            opt_sz.append(1/max(eig_vals))    
        return opt_sz

    def get_dual_Hessian(self, edges_to_shared_node):
        node_Hessians = [ (self.M/2) * np.linalg.inv(self.X[i].T @ self.X[i]) for i in range(self.n) ]
        Hessian = [[None for l in range(self.E)] for m in range(self.E)] # Hessian[i,k] = Hessian[k,i]
        for l in range(self.E):
            for m in range(l+1): # l and m can take the same value 
                if (l != m) and edges_to_shared_node[l,m] > -1: # if edges l and m share a node
                    i = edges_to_shared_node[l,m]
                    Hessian[l][m] = (-1) * self.A[l,i] * self.A[m,i] * node_Hessians[i]
                    Hessian[m][l] = (-1) * self.A[l,i] * self.A[m,i] * node_Hessians[i]
                elif l == m:
                    i, j = self.edge_to_nodes[l]
                    Hessian[l][l] = (-1) * (node_Hessians[i] + node_Hessians[j])
        return Hessian



# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class centralized_LR():
    """
    Centralized LR for computing the optimal solution taking into account all data 
    . Y lebels assumed to be already 1 and -1
    """
    def __init__(self, X, y, c):
        # Parameters
        self.X = np.c_[np.ones((X.shape[0], 1)), X] # add ones for bias term
        self.y = y # assumed to be already 1 and -1
        self.c = c
        self.M = X.shape[0]
        self.d = self.X.shape[1]

    def objective(self, theta):
        obj = (1/self.M) * np.sum( np.log(1 + np.exp(-self.y * np.dot(self.X, theta)) ) ) + self.c * theta @ theta
        return obj

    def df_objective(self, theta):
        factor_1 = -self.y[:, np.newaxis] @ np.ones((1,self.d)) * self.X
        arg = -self.y * np.dot(self.X, theta)
        factor_2 = np.exp(arg) / (1 + np.exp(arg))
        df_obj = (1/self.M) * (factor_1.T @ factor_2).flatten() + 2 * self.c * theta
        return df_obj 

    def get_optimal_params(self,):
        # calling scipy for the optimization
        theta = np.zeros((self.d, 1))
        output = fmin_tnc(func=self.objective, x0=theta, fprime=self.df_objective)
        parameters = output[0]
        code = output[2]
        return parameters, code



# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class LR_Problem_2():
    """
    Logistic Regression problem with the cost
    f(u) =
    give all nodes some samples of both classes 
    """
    def __init__(self, n, A, dim, edge_to_nodes):
        # Parameters
        self.n = n
        self.d = dim # accounts for the bias, so we call make_classification with self.d - 1
        self.A = A
        self.E = np.shape(A)[0]
        self.edge_to_nodes = edge_to_nodes

        self.c = 0.1 # weight of the regularizer

        # Generate data - for now we fix results to dimension = 2
        self.M = 50
        Xall, Yall = make_classification(n_samples=self.M*self.n, n_features=self.d-1, n_informative=self.d-1, n_redundant=0, \
            n_repeated=0, n_classes=2, n_clusters_per_class=1, flip_y=0, shuffle=False, random_state=2250)
        Yall = 2*Yall - 1 # make labels 1 and -1
        Xall_e = np.c_[np.ones((Xall.shape[0], 1)), Xall] # add ones for bias term
        self.X = [None] * n
        self.Y = [None] * n
        for i in range(n):
            half_samp = int(self.M*self.n / 2)
            start_idx = i*int(self.M/2)
            end_idx = (i+1)*int(self.M/2)            
            self.X[i] = np.vstack( ( Xall_e[ start_idx : end_idx ,:] , Xall_e[ half_samp+start_idx : half_samp+end_idx ,:] ) )
            self.Y[i] = np.concatenate( ( Yall[ start_idx : end_idx ] , Yall[ half_samp+start_idx : half_samp+end_idx ] ) )

        # Get optimal solution for the complete data with 
        central_LR = centralized_LR(Xall, Yall, self.c)
        self.theta_opt, output_code = central_LR.get_optimal_params()
        theta_opt_tile = np.tile( self.theta_opt.reshape((1,self.d)), (n,1) )
        self.analy_opt_obj_val = self.objective( theta_opt_tile ) # objective value for optimal theta

    def sigmoid(self, x_vec):  # CONSUMES TOO MUCH TIME
        """
        Numerically stable sigmoid function.
        Taken from: 
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        """
        z_vec = np.zeros(np.shape(x_vec))
        for i, x in enumerate(x_vec):
            if x >= 0:
                z = math.exp(-x)
                z_vec[i] = 1 / (1 + z)
            else:
                # if x is less than zero then z will be small, denom can't be
                # zero because it's 1+z.
                z = math.exp(x)
                z_vec[i] = z / (1 + z)
        return z_vec

    def objective(self, thetas):
        loss_per_node = []
        for i in range(self.n): 
            arg = -self.Y[i] * np.dot(self.X[i], thetas[i,:]) 
            loss_list = [ a if a > 1e2 else np.log(1 + np.exp(a)) for a in arg] # CONSUMES TOO MUCH TIME
            loss_per_node.append( (1/self.M) * sum(loss_list) + self.c * thetas[i,:] @ thetas[i,:] )
        obj_val = sum(loss_per_node)
        return obj_val

    def per_node_Lagrangian(self, theta, lambdas, i):
        arg = -self.Y[i] * np.dot(self.X[i], theta) 
        loss_list = [ a if a > 1e2 else np.log(1 + np.exp(a)) for a in arg] # CONSUMES TOO MUCH TIME
        loss = (1/self.M) * sum(loss_list) 
        regularizer =  self.c * theta @ theta
        relaxation = (self.A[:,i] @ lambdas) @ theta
        result = loss + regularizer + relaxation
        return result 

    def df_per_node_Lagrangian(self, theta, lambdas, i):
        df_loss_1 = -self.Y[i][:, np.newaxis] @ np.ones((1,self.d)) * self.X[i] # [M x d]
        arg = -self.Y[i] * np.dot(self.X[i], theta)  # [M x 1]
        df_loss_2 = self.sigmoid(arg) # [M x 1]
        df_regularizer = 2 * self.c * theta # [d x 1]
        df_relaxation = (self.A[:,i] @ lambdas)  # [d x 1]
        result =  (1/self.M) * (df_loss_1.T @ df_loss_2) + df_regularizer + df_relaxation
        return result

    def arg_min_Lagran(self, i, lambdas): # PER NODE (called from the solvers)
        theta_0 = np.zeros((lambdas.shape[1],))
        output = fmin_tnc(func=self.per_node_Lagrangian, x0=theta_0, fprime=self.df_per_node_Lagrangian, args=(lambdas, i))
        return output[0]

    def Lagrangian(self, thetas, lambdas): # same as in the other problems 
        # As written here, problem-independent
        Lag_val = self.objective(thetas) + np.matrix.trace(lambdas.T @ self.A @ thetas)
        return Lag_val





# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class LR_Problem():
    """
    Logistic Regression problem with the cost
    f(u) =
    give all nodes some samples of both classes 
    """
    def __init__(self, n, A, dim, edge_to_nodes):
        # Parameters
        self.n = n
        self.d = dim # accounts for the bias, so we call make_classification with self.d - 1
        self.A = A
        self.E = np.shape(A)[0]
        self.edge_to_nodes = edge_to_nodes

        self.c = 0.1 # weight of the regularizer

        # Generate data - for now we fix results to dimension = 2
        self.M = 50
        Xall, Yall = make_classification(n_samples=self.M*self.n, n_features=self.d-1, n_informative=self.d-1, n_redundant=0, \
            n_repeated=0, n_classes=2, n_clusters_per_class=1, flip_y=0, shuffle=False, random_state=2250)
        Yall = 2*Yall - 1 # make labels 1 and -1
        Xall_e = np.c_[np.ones((Xall.shape[0], 1)), Xall] # add ones for bias term
        self.X = [None] * n
        self.Y = [None] * n
        for i in range(n):
            half_samp = int(self.M*self.n / 2)
            start_idx = i*int(self.M/2)
            end_idx = (i+1)*int(self.M/2)            
            self.X[i] = np.vstack( ( Xall_e[ start_idx : end_idx ,:] , Xall_e[ half_samp+start_idx : half_samp+end_idx ,:] ) )
            self.Y[i] = np.concatenate( ( Yall[ start_idx : end_idx ] , Yall[ half_samp+start_idx : half_samp+end_idx ] ) )

        # Get optimal solution for the complete data with 
        central_LR = centralized_LR(Xall, Yall, self.c)
        self.theta_opt, output_code = central_LR.get_optimal_params()
        theta_opt_tile = np.tile( self.theta_opt.reshape((1,self.d)), (n,1) )
        self.analy_opt_obj_val = self.objective( theta_opt_tile ) # objective value for optimal theta

    def sigmoid(self, x):
        """
        Numerically stable sigmoid function.
        Taken from: 
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        """        
        z = np.zeros(np.shape(x))
        flag_more = x >= 0
        flag_less = x < 0
        z[flag_more] = 1 / (1 + np.exp(-x[flag_more]))
        z[flag_less] = np.exp(x[flag_less]) / (1 + np.exp(x[flag_less]))
        return z

    def objective(self, thetas):
        loss_per_node = []
        for i in range(self.n): 
            arg = -self.Y[i] * np.dot(self.X[i], thetas[i,:]) 
            # loss_list = [ a if a > 1e2 else np.log(1 + np.exp(a)) for a in arg] # CONSUMES TOO MUCH TIME
            loss_list = np.zeros(np.shape(arg)) 
            flag_more = arg >= 1e2
            flag_less = arg < 1e2
            loss_list[flag_more] = arg[flag_more]
            loss_list[flag_less] = np.log(1 + np.exp(arg[flag_less]))

            loss_per_node.append( (1/self.M) * sum(loss_list) + self.c * thetas[i,:] @ thetas[i,:] )
        obj_val = sum(loss_per_node)
        return obj_val

    def per_node_Lagrangian(self, theta, lambdas, i):
        arg = -self.Y[i] * np.dot(self.X[i], theta) 

        loss_list = np.zeros(np.shape(arg)) 
        flag_more = arg >= 1e2
        flag_less = arg < 1e2
        loss_list[flag_more] = arg[flag_more]
        loss_list[flag_less] = np.log(1 + np.exp(arg[flag_less]))

        loss = (1/self.M) * sum(loss_list) 
        regularizer =  self.c * theta @ theta
        relaxation = (self.A[:,i] @ lambdas) @ theta
        result = loss + regularizer + relaxation
        return result

    def df_per_node_Lagrangian(self, theta, lambdas, i):
        df_loss_1 = -self.Y[i][:, np.newaxis] @ np.ones((1,self.d)) * self.X[i] # [M x d]
        arg = -self.Y[i] * np.dot(self.X[i], theta)  # [M x 1]
        df_loss_2 = self.sigmoid(arg) # [M x 1]
        df_regularizer = 2 * self.c * theta # [d x 1]
        df_relaxation = (self.A[:,i] @ lambdas)  # [d x 1]
        result =  (1/self.M) * (df_loss_1.T @ df_loss_2) + df_regularizer + df_relaxation
        return result

    def arg_min_Lagran(self, i, lambdas): # PER NODE (called from the solvers)
        theta_0 = np.zeros((lambdas.shape[1],))
        output = fmin_tnc(func=self.per_node_Lagrangian, x0=theta_0, fprime=self.df_per_node_Lagrangian, args=(lambdas, i))
        return output[0]

    def Lagrangian(self, thetas, lambdas): # same as in the other problems 
        # As written here, problem-independent
        Lag_val = self.objective(thetas) + np.matrix.trace(lambdas.T @ self.A @ thetas)
        return Lag_val

