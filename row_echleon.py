
import sympy as sympy 
from sympy import GF
from sympy.polys.matrices import DomainMatrix
import numpy as np
import galois

def len_unique_elements(arr):
    return len(set(arr))

def get_H_arr(dv, dc, k, n):
    """ Gets Tanner Graph Connections for each Variable Node """

    # Initialize an array of size dv*n and fill it with numbers for each variable node's connections
    arr = np.arange(0, dv*n)
    flag = 0

    while True:
        flag = 0

        # Generate a random permutation of the array
        arr = np.random.permutation(arr)

        # Checking if each check node is connected to dv variable nodes
        t = [arr[i:i+dv] for i in range(0, len(arr), dv)]
        for i in t:
            i = i//dc
            if len_unique_elements(i) != dv:
                flag +=1
        
        # Break if all check nodes are connected to dv variable nodes
        if flag == 0:
            break   

    return arr

def get_H_Matrix(dv, dc, k, n, Harr=None):
    """ Creates the H matrix from the Variable Node Connections of the Tanner Graph """
    
    if Harr is None:
        Harr = get_H_arr(dv, dc, k, n)

    # Initialize H matrix - the size is wrong will need to fix at some point
    H = np.zeros((n-k, n))

    # Fill H matrix where Variable Node is connected to Check Node
    for (i,j) in enumerate(Harr):
        H[j//dc, i//dv] = 1

    return H

def get_H_matrix_sclpdc(dv, dc, k, n, Harr):
    """ Creates the H matrix from the Variable Node Connections of the  variable Tanner Graph """
    
    if Harr is None:
        Harr = get_H_arr(dv, dc, k, n)

    # Initialize H matrix - the size is wrong will need to fix at some point
    H = np.zeros((n-k, n))

    dv = dv[0]
    # Fill H matrix where Variable Node is connected to Check Node
    for (i,j) in enumerate(Harr):
        H[j, i//dv] = 1
    
    return H


def get_reduced_row_echleon_form_H(H, ffdim=2):
    """ Returns the reduced row echleon form of H """

    # Using the Domain Matrice Instead
    ff = GF(ffdim)
    H = DomainMatrix([[ff(H[i,j]) for j in range(H.shape[1])] for i in range(H.shape[0])], H.shape, ff)
    H_rref = H.rref()[0]
    return np.array(H_rref.to_Matrix())

def check_standard_form_variance(H):
    """ Checks if the H matrix is in standard form and returns columns that need changing """
    n = H.shape[1]
    k = n - H.shape[0]
    shape = H.shape
    I_dim = n-k 
    I = np.eye(I_dim)
    columns_to_change = {}
    rows = shape[1]
    print(H.shape)
    
    # Check if the last I_dim columns are I
    if np.all(H[:,k:n] == I):
        return None
    else:
        # Find the columns that need changing
        for i in range(k, n):
            if not np.all(H[:,i] == I[:,i-k]):
                columns_to_change[i] = I[:,i-k]
    
    return columns_to_change

def switch_columns(H, columns_to_change):
    """ Finds and makes the necessary column switches to convert H to standard form """
    
    # If no columns need changing, return
    if not columns_to_change:
        return H, None

    n = H.shape[1]
    k = n - H.shape[0]
    column_positions = list(columns_to_change.keys())
    changes_made = []
    switches = []
    I = np.eye(n-k)
    
    for i in column_positions:
        for j in range(n):
            if np.all(H[:,j] == I[:, i-k]):
                if j in changes_made:
                    continue
                changes_made.append(i)
                switches.append((i,j))
                t = H[:,i].copy()
                H[:,i] = H[:,j]
                H[:,j] = t
                break

    return H, switches

def standard_H_to_G(H, ffdim=2, switches = None):
    """ Inverts the standard H matrix to get the Generator Matrix"""
    n = H.shape[1]
    k = n - H.shape[0]
    P = -H[:,0:k]
    G = np.hstack((np.eye(k), P.T)) % ffdim
    
    if switches is None:
        return G.astype(int)
    
    # Since switches made forward, need to reverse list to undo
    switches = list(reversed(switches))
    if switches: 
        for i in switches:
            t = G[:,i[0]].copy()
            G[:,i[0]] = G[:,i[1]]
            G[:,i[1]] = t

    return G.astype(int)


def parity_to_generator(H, ffdim=2):
    """ Converts a parity check matrix to a generator matrix """
    H_rref = get_reduced_row_echleon_form_H(H, ffdim=ffdim)
    H_st, switches = switch_columns(H_rref, check_standard_form_variance(H_rref))
    G = standard_H_to_G(H_st, switches=switches, ffdim=ffdim)
    
    return G

def alternative_parity_to_generator(H, ffdim=2):
    """ Converts a parity check matrix to a generator matrix using it's nullspace"""
    
    GF = galois.GF(ffdim)
    t = GF(H.astype(int))
    return np.array(t.null_space())

