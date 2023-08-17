"""
Functions of illustrative_problem.ipynb

28/01/2022
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.linalg import sqrtm
import time
import networkx as nx
import json
from datetime import datetime



# GRAPHS
def clique_graph(n):
    G = np.ones((n,n))
    A = G - np.diag(np.ones((n,)))
    return A

def random_graph(n,p):
    G = nx.fast_gnp_random_graph(n,p)
    A = np.array( nx.to_numpy_array(G) )
    return A

def power_law_graph(n,m):
    G = nx.barabasi_albert_graph(n, m)
    A = np.array( nx.to_numpy_array(G) ).astype(int)
    return A

def watts_strogatz_graph(n,d,p):
    G = nx.watts_strogatz_graph(n,d,p)
    A = np.array( nx.to_numpy_array(G) ).astype(int)
    return A

def grid_graph(n):
    G = nx.grid_graph(dim=[n,n])
    A = np.array(nx.to_numpy_array(G))
    return A

def d_regular_random_graph(n,d):
    G = nx.random_regular_graph(d, n)
    A = np.array( nx.to_numpy_array(G) ).astype(int)
    return A

def geographic_graph(n,radius):
    node_pos = np.random.uniform(low=0, high=1, size=(n,2))
    dist_mat = np.zeros((n,n))
    for c in range(n): # go through columns
        for r in range(c): # go through columns
            dist_mat[r][c] = np.linalg.norm(node_pos[r,:] - node_pos[c,:])
            dist_mat[c][r] = dist_mat[r][c]
    dist_mat = dist_mat + np.eye(n)
    G = (dist_mat < radius).astype(int)
    return G, node_pos

def tree_graph(branching_factor, depth):
    B = nx.balanced_tree(branching_factor, depth)
    G = np.array(nx.to_numpy_array(B))
    return G

def linked_binary_trees_graph(n):
    # NOTE: we don't use n for this graph
    branching_factor = 3
    depth = 2
    tree = tree_graph(branching_factor, depth) # already np array
    link_mat = np.zeros(np.shape(tree))
    link_mat[0,0] = 1 # root is the first node
    G = np.block([[tree, link_mat], [link_mat.T, tree]]) # connect both trees through their root
    return G

def make_graph(n, graph_name, *args):
    max_graph_apptempts = 100 # maximum number of tries allowed to generate a connected graph
    if graph_name == 'clique':
        graph_fun = clique_graph
    elif graph_name == 'random':
        graph_fun = random_graph
    elif graph_name == 'power-law':
        graph_fun = power_law_graph
    elif graph_name == 'WS':
        graph_fun = watts_strogatz_graph
    elif graph_name == 'grid':
        graph_fun = grid_graph
    elif graph_name == 'd-regular':
        graph_fun = d_regular_random_graph
    elif graph_name == 'geographic':
        graph_fun = geographic_graph
    elif graph_name == 'tree':
        graph_fun = tree_graph
    elif graph_name == 'linked_trees':
        graph_fun = linked_binary_trees_graph
    else:
        raise Exception("graph name is not valid")
    G = graph_fun(n,*args)
    if graph_name == 'geographic':
        node_pos = G[1]
        G = G[0]
    connected, _ = is_connected(G)
    counter = 0
    while not connected:
        counter += 1
        G = graph_fun(n,*args)
        if graph_name == 'geographic':
            node_pos = G[1]
            G = G[0]
        connected, _ = is_connected(G)
        if counter == max_graph_apptempts:
            raise Exception('Could not generate connected graph in ' + str(max_graph_apptempts) + ' attempts')
            return
    if graph_name == 'geographic':
        G = (G, node_pos)
    return G



# ..........................................................
def is_connected(A):
    import numpy as np
    # Compute Laplacian to test if it is connected
    d = A.sum(axis=1)
    D = np.diag(d)
    L = D - A
    eig_L, _ = np.linalg.eig(L)
    n_connected_components = np.sum(eig_L < 1e-10)
    if n_connected_components == 1:
        connected = True
    else:
        connected = False
    return connected, n_connected_components



# ..........................................................
def get_edge_matrix(A):
    # Matrix of zeros with {1,-1} indicating each edge (direction is first node --> last node)
    # For N nodes with E edges returns an [E x N] matrix
    A = np.where(A != 0, 1, 0)
    A_UT = np.triu(A, k=0)
    e = np.sum(np.sum(A_UT)) # number of edges
    n = np.shape(A)[0] # number of nodes
    E = np.zeros((e,n))
    coords_edges = np.nonzero(A_UT)
    E[range(e),coords_edges[0]] = 1 # first node, or leader
    E[range(e),coords_edges[1]] = -1 # second node, or follower
    return E



# ..........................................................
def create_node_variables(A):

    n = np.shape(A)[1]

    role = [None] * n # stores for each node a vector fo the length of its neighbors stating whether the node is leader (1) or follower (-1)
    neighbors = [None] * n # states for each node who are its neighbors
    N = [None] * n # number of neighbors of each node
    nodes_edges = [None] * n # indeces of edges connected to node each node

    for i in range(n):
        edges_i = A[A[:,i]!=0,:] # edges concerning node i (selects rows of A)
        role[i] = np.expand_dims( np.copy(edges_i[:,i]), axis=1 ) # +1 or -1, tells us if node i is leader of follower of the edge
        edges_i[:,i] = 0 # delete values column i
        neighbors[i] = np.nonzero(edges_i)[1] # save indeces of neighbor nodes (column indeces of non-zero entries edges_i)
        N[i] = len(neighbors[i])
        edge_indicator = A[:,i]!=0
        nodes_edges[i] = [idx for idx, value in enumerate(edge_indicator) if value]
    return role, neighbors, N, nodes_edges



# ..........................................................
def create_edge_variables(A,neighbors):

    n = np.shape(A)[1]
    E = np.shape(A)[0]

    mut_idcs = np.zeros((n,n)).astype(int) # save mutual indeces in matrix form: (row i, col j) holds the index of node j for node i
    mat_edge_idxs = np.zeros((n,n)).astype(int)
    edge_to_nodes = [None] * E

    for idx_row, row in enumerate(A):

        i = np.where(row == 1)[0][0]
        j = np.where(row == -1)[0][0]

        idx_j = np.where(neighbors[i] == j)[0][0]
        idx_i = np.where(neighbors[j] == i)[0][0]

        mat_edge_idxs[i][j] = idx_row
        mat_edge_idxs[j][i] = idx_row

        mut_idcs[i][j] = idx_j
        mut_idcs[j][i] = idx_i

        edge_to_nodes[idx_row] = (i,j)

    return mut_idcs, mat_edge_idxs, edge_to_nodes



# ..........................................................
def get_edges_to_shared_node(A, edge_to_nodes):
    E = np.shape(A)[0]
    AA = abs(A @ A.T) # [E x E]
    edges_to_shared_node = -np.ones((E,E), dtype=int)
    for l in range(E):
        for m in range(l):
            if AA[l,m] > 0: # if edges l and m share a node
                L = set(edge_to_nodes[l])
                M = set(edge_to_nodes[m])
                edges_to_shared_node[l,m] = list(L & M)[0]
                edges_to_shared_node[m,l] = list(L & M)[0]
    return edges_to_shared_node



# ..........................................................
def save_data(data, origin_script):
    now_in_string = str(datetime.now())
    date_time = now_in_string.split()
    year_month_day = date_time[0].split('-')
    hs_mins_secs = date_time[1].split(':')
    save_name = 'results/'+origin_script+'-M'+year_month_day[1]+'D'+year_month_day[2]+'_'+hs_mins_secs[0]+'h'+hs_mins_secs[1]+'m.json'
    with open(save_name, "w") as write_file:
        json.dump(data, write_file)
    print('\n Results saved with name: \t', save_name, '\n\n')
    return save_name
