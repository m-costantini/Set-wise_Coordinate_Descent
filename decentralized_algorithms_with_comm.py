"""
Identical to decentralized_algorithms.py but it additionally returns the number of communication rounds for comparison in the experiment of Section VII.C
"""

import numpy as np
from random import choice
import sys, time

class SxCD():
    """
    Available algorithms:
    SU-CD       Uniform neighbor selection
    SGS-CD      Gauss-Southwell neighbor selection
    SL-CD       Random neighbor selection with Lipschitz probabilities
    SGSL-CD     Gauss-Southwell-Lipschitz neighbor selection
    SeL-CD      Random neighbor selection with Lipschitz probabilities and estimated constants
    SGSeL-CD    Gauss-Southwell-Lipschitz neighbor selection with estimated constants

    -------------------------------------------------
    Computation of the communication rounds:

    SU-CD       2: one vector per node
    SGS-CD      deg + 1: all the neighbors + the one of the node to the neighbor chosem 
    SL-CD       2: as SU-CD, but with different probabilities
    SGSL-CD     deg + 1: as SGS-CD, but with different computation
    SeL-CD      2 + iter*2: 2 from first exchange before entering do-while, extra 2 per each do-while pass
    SGSeL-CD    deg + 1 + iter*2: from combining the previous

    """
    def __init__(self, the_problem, simu_vars, solver_name):

        self.the_problem = the_problem
        self.solver_name = solver_name
        # parameter unpacking
        self.dim = simu_vars['dimensions']
        self.n = simu_vars['n']
        self.E = simu_vars['E']
        self.A = simu_vars['A']
        self.role = simu_vars['role']
        self.neighbors = simu_vars['neighbors']
        self.N = simu_vars['N']
        self.nodes_edges = simu_vars['nodes_edges']
        self.mat_edge_idxs = simu_vars['mat_edge_idxs']
        self.edge_to_nodes = simu_vars['edge_to_nodes']
        self.stepsizes = simu_vars['stepsizes']
        self.mut_idcs = simu_vars['mut_idcs']
        self.steps = simu_vars['steps']
        self.iter_subsamp = simu_vars['iteration_subsampling']
        # variables initialization
        self.thetas = 10*np.ones(shape=(self.n,self.dim))
        self.lambdas = 10*np.ones(shape=(self.E,self.dim))      
        
        self.obj = []
        self.dual = []
        self.comm_rounds = []

        if (self.solver_name == 'SeL-CD') or (self.solver_name == 'SGSeL-CD'): # initialize Lipschitz constants
            self.Lest = [1e-2 for e in range(self.E)] # list of ESTIMATED Lipschitz constants
        elif (self.solver_name == 'SL-CD') or (self.solver_name == 'SGSL-CD'): # compute Lipschitz constants
            self.L = self.the_problem.L # REAL Lipschitz constants
            for e in range(self.E):
                self.stepsizes[e] = 1/self.L[e]

    def choose_neighbor(self, i):
        if self.solver_name == 'SU-CD':
            idx_j = np.random.randint(self.N[i]) # random uniform selection
        elif self.solver_name == 'SeL-CD':
            Lest_of_i = [self.Lest[e] for e in self.nodes_edges[i]]
            idx_j = np.random.choice(np.arange(0,self.N[i]), p=[Lj/sum(Lest_of_i) for Lj in Lest_of_i]) # proportional to Lipschitz value
        elif self.solver_name == 'SL-CD':
            Ls_of_i = [self.L[e] for e in self.nodes_edges[i]]
            idx_j = np.random.choice(np.arange(0,self.N[i]), p=[Lj/sum(Ls_of_i) for Lj in Ls_of_i]) # proportional to Lipschitz value
        elif (self.solver_name == 'SGS-CD') or (self.solver_name == 'SGSL-CD') or (self.solver_name == 'SGSeL-CD'):
            # NOTE: we unnecesarily repeat the computation of theta_i and theta_j afterwards for this algorithm, but it's the price of recycling code
            theta_i = self.the_problem.arg_min_Lagran(i, self.lambdas).flatten() # compute theta of activated node
            # ask the parameters of all of our neighbors to compute the largest gradient
            theta_neighs = np.zeros((self.N[i],self.dim))
            for idx_k, k in enumerate(self.neighbors[i]): # weighted already by the corresponding A[e,k]
                theta_neighs[idx_k,:] = self.the_problem.arg_min_Lagran(k, self.lambdas).flatten()
            # compute magnitude of the gradients and choose neighbor that has the largest
            grads_i = self.role[i] @ np.expand_dims(theta_i, axis=0) - np.tile(self.role[i],(1,self.dim)) * theta_neighs
            mag_grads_i = np.linalg.norm(grads_i, axis=1)
            # apply algorithm's rule
            if self.solver_name == 'SGS-CD':
                idx_j = np.where( mag_grads_i == np.max(mag_grads_i) )[0]
            elif self.solver_name == 'SGSL-CD':
                Ls_of_i = np.array([self.L[e] for e in self.nodes_edges[i]])
                scaled_mag_grads_i = mag_grads_i / np.sqrt(Ls_of_i)
                idx_j = np.where( scaled_mag_grads_i == np.max(scaled_mag_grads_i) )[0]
            else: # SGSeL-CD
                Lest_of_i = np.array([self.Lest[e] for e in self.nodes_edges[i]])
                scaled_mag_grads_i = mag_grads_i / np.sqrt(Lest_of_i)
                idx_j = np.where( scaled_mag_grads_i == np.max(scaled_mag_grads_i) )[0]
            # tie breaking
            if hasattr(idx_j, "__len__"): # random unifrom choice among possibilities
                idx_j = choice(idx_j)
        else:
            raise Exception("Invalid solver name")
        j = self.neighbors[i][idx_j]
        e = self.mat_edge_idxs[i,j]
        idx_i = self.mut_idcs[j][i]
        return idx_j, j, e, idx_i


    def coordinate_step_with_Lips_estimation(self, i, j, idx_i, idx_j, e, grad_e):
        if self.solver_name == 'SeL-CD':
            self.comm += 2
        else: # i.e. solver = SGSeL-CD
            self.comm += self.N[i] + 1
        lambda_e_0 = np.copy(self.lambdas[e,:])
        L_i = 0.001 # old ---> works better than the one below
        grad_new = -grad_e # to make sure that we enter the loop below
        while grad_e @ grad_new < 0:
            L_i = 2*L_i
            self.lambdas[e,:] = lambda_e_0 + (1/L_i) * grad_e
            theta_i = self.the_problem.arg_min_Lagran(i, self.lambdas).flatten()
            theta_j = self.the_problem.arg_min_Lagran(j, self.lambdas).flatten()
            grad_new = self.A[e,i]*theta_i + self.A[e,j]*theta_j
            self.comm += 2
        self.Lest[i] = 0.5 * L_i # new 
        return


    def solve(self,):
        comm_cumul = 0
        for t in range(self.steps):
            self.comm = 0
            print(f"\rCompleted {int((t+1)/self.steps*100)}%", end='') # print progress
            # activated node
            i = np.random.randint(self.n) # random node goes active
            idx_j, j, e, idx_i = self.choose_neighbor(i) # choose contacted neighbor
            # compute primal variables that minimize the Lagrangian
            theta_i = self.the_problem.arg_min_Lagran(i, self.lambdas).flatten()
            theta_j = self.the_problem.arg_min_Lagran(j, self.lambdas).flatten()
            self.thetas[i,:] = theta_i
            self.thetas[j,:] = theta_j
            # update lambdas
            grad = self.A[e,i]*theta_i + self.A[e,j]*theta_j
            if (self.solver_name == 'SeL-CD') or (self.solver_name == 'SGSeL-CD'):
                self.coordinate_step_with_Lips_estimation(i, j, idx_i, idx_j, e, grad) # coordinate step with stepsize search
            else: # SU-CD, SL-CD, SGS-CD, SGSL-CD
                self.lambdas[e,:] = self.lambdas[e,:] + self.stepsizes[e] * grad
                if (self.solver_name == 'SU-CD') or (self.solver_name == 'SL-CD'):
                    self.comm += 2
                else: # SGS-CD or SGSL-CD
                    self.comm += self.N[i] + 1
            comm_cumul += self.comm
            # compute primal and dual function value
            if t % self.iter_subsamp == 0: # improve memory use
                self.obj.append(float(self.the_problem.objective(self.thetas)))
                self.dual.append(float(self.the_problem.Lagrangian(self.thetas, self.lambdas)))
                self.comm_rounds.append(comm_cumul)
                # check divergence / convergence conditions
                last_suboptim_val = abs(1 - self.dual[-1]/self.the_problem.analy_opt_obj_val)
                if last_suboptim_val > 10**20: # thing is diverging !
                    print(' --> Objective > breaking_thresh @ iter ',t ,' --> break')
                    break
                elif last_suboptim_val < 10**(-9): # converged! stop!
                    print(' --> Precision < 10^(-9) reached @ iter',t ,' --> leave!')
                    break
        print("                                                                               ", end='\r') # delete progress printing
        return np.array(self.obj), np.array(self.dual), self.comm_rounds

