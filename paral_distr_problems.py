import numpy as np

class QuadraticProblem():
    def __init__(self, coord_sets, P):
        
        # super(Distrib_problem, self).__init__()
        self.coord_sets = coord_sets
        self.P = P
        
        self.opt_val = 1

        # Pre-compute the matrices to compute the gradients per set
        PP = P + P.T # function gradient 
        self.set_PP = []
        for c_set in coord_sets:
            grad = PP[c_set,:][:,c_set]
            self.set_PP.append( grad )

    # ---------------------------------------------------------------------------------- 
    # Generate functions
        
    def objective(self, x):
        return x.T @ self.P @ x + self.opt_val

    def get_set_grads(self, x, i):
        x_set = x[self.coord_sets[i]]
        return self.set_PP[i] @ x_set
