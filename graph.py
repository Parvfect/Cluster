
import numpy as np
import row_echleon as r
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from cProfile import Profile
from pstats import Stats
import re
import sys

def permuter(arr, ffield, vn_value):

    possibilities = set(arr[0])
    new_possibilities = set()
    for i in range(1, len(arr)):
        for k in possibilities:
            for j in arr[i]:
                new_possibilities.add((j+k) % ffield)
                if len(new_possibilities) == ffield:
                    return vn_value
        possibilities = new_possibilities 
        new_possibilities = set()
    
    return {(-p)%ffield for p in possibilities}

def conv_circ(u, v):
    """Perform circular convolution between u and v over GF using FFT."""
    return np.real(np.fft.ifft(np.fft.fft(u) * np.fft.fft(v)))

def perform_convolutions(arr_pd):
    """ Combines all the Probability distributions within the array using the Convolution operator
    
    Args:
        arr_pd (arr): Array of Discrete Probability Distributions
    
    Returns:
        conv_pd (arr): Combined Probability Distributions after taking convolution over all of the pdf
    """

    pdf = arr_pd[0]

    for i in arr_pd[1:]:
        pdf = conv_circ(pdf, i)

    return pdf

class Node:

    def __init__(self, no_connections, identifier):
        self.value = 0
        self.links = np.zeros(no_connections, dtype=int)
        self.identifier = identifier

    def add_link(self, node):
        """ Adds a link to the node. Throws an error if the node is full """
        
        # Check if node is full
        #if np.all(self.links):
         #   raise ValueError("Node is full")

        # Add to empty link 
        for (i,j) in enumerate(self.links):
            if not j:
                self.links[i] = node.identifier
                break
        
        return self.links
    
    def replace_link(self, node, index):
        """ Replaces a link with another link """
        self.links[index] = node
        return self.links
    
class CheckNode(Node):

    def __init__(self, dc, identifier):
        super().__init__(dc, identifier)
    
class VariableNode(Node):
    def __init__(self, dv, identifier):
        super().__init__(dv, identifier)

class TannerGraph:
    """ Initializes empty, on establishing connections creates H and forms links """

    def __init__(self, dv, dc, k, n, ffdim=2):

        self.vns = [VariableNode(dv, i) for i in range(n)]
        self.cns = [CheckNode(dc, i) for i in range(n-k)]
        self.dv = dv
        self.dc = dc
        self.k = k
        self.n = n
        self.ffdim = ffdim

    def establish_connections(self, Harr=None):
        """ Establishes connections between variable nodes and check nodes """
        
        # In case Harr is sent as a parameter
        if Harr is None:
            self.Harr = r.get_H_arr(self.dv, self.dc, self.k, self.n)
        else:
            self.Harr = np.array(Harr)
        
        Harr = self.Harr//self.dc

        # Divide Harr into dv parts
        Harr = [Harr[i:i+self.dv] for i in range(0, len(Harr), self.dv)]

        # Establish connections
        for (i,j) in enumerate(Harr):
            for k in j:
                self.vns[i].add_link(self.cns[k])
                self.cns[k].add_link(self.vns[i])


    def get_connections(self):
        """ Returns the connections in the Tanner Graph """
        return [(i.identifier, j) for i in self.cns for j in i.links]

    def get_cn_link_values(self, cn):
        """ Returns the values of the connected vns for the cn as a dd array"""
        vals = []
        for i in cn.links:
            vals.append(self.vns[i].value)

        return vals
    
    def assign_values(self, arr):   

        assert len(arr) == len(self.vns) 

        for i in range(len(arr)):
            self.vns[i].value = arr[i]

    def coupon_collector_erasure_decoder(self, max_iterations=100):
        """ Belief Propagation decoding for the general case (currently only works for BEC) )"""

        unresolved_vns = sum([1 for i in self.vns if len(i.value) > 1 ])
        resolved_vns = 0
        prev_resolved_vns = 0
        
        for iteration in range(max_iterations):
            
            for i in self.cns:
                for j in i.links:
                    sum_vns = 0
                    uncertainty_check = False
                    
                    for k in i.links:
                        if k != j:
                            if not type(self.vns[k].value) == int:
                                if len(self.vns[k].value) > 1:
                                    uncertainty_check = True
                                    break
                            if type(self.vns[k].value) == int:
                                sum_vns += self.vns[k].value    
                            else:
                                sum_vns += self.vns[k].value[0]
                    
                    if uncertainty_check:
                        continue
                    
                    if len(self.vns[k].value) > 1:
                        resolved_vns += 1    
                    
                    self.vns[j].value = [-sum_vns % self.ffdim]  

                if unresolved_vns == resolved_vns:
                    return np.array([i.value for i in self.vns])
            
            if prev_resolved_vns == resolved_vns:
                    return np.array([i.value for i in self.vns])
            prev_resolved_vns = resolved_vns
            
        return np.array([i.value for i in self.vns])


    def coupon_collector_decoding(self, max_iterations=10000):
        """ Decodes for the case of symbol possiblities for each variable node 
            utilising Belief Propagation - may be worth doing for BEC as well 
        """
        
        unresolved_vns = sum([1 for i in self.vns if len(i.value) > 1 ])
        resolved_vns = 0
        total_possibilites = sum([len(i.value) for i in self.vns])
        
        while True:
            # Iterating through all the check nodes
            for i in self.cns:
                
                vn_vals = self.get_cn_link_values(i)
                
                for j in i.links:
                
                    vals = vn_vals.copy()
                    current_value = self.vns[j].value
                    vals.remove(current_value)

                    possibilites = permuter(vals, self.ffdim, current_value)
                    new_values = set(current_value).intersection(set(possibilites))
                    self.vns[j].value = list(new_values)
                    
                    if len(current_value) > 1 and len(new_values) == 1:
                        resolved_vns += 1
                    
                decoded_values = [i.value for i in self.vns]

                if unresolved_vns ==  resolved_vns and sum([len(i) for i in decoded_values]) == len(decoded_values):
                    return np.array([i.value for i in self.vns])
            
            if sum([len(i.value) for i in self.vns]) == total_possibilites:
                return [i.value for i in self.vns]
            
            total_possibilites = sum([len(i.value) for i in self.vns])
            
            prev_resolved_vns = resolved_vns   
        
        return [i.value for i in self.vns]
