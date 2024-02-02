

import numpy as np
import matplotlib.pyplot as plt
import time
import row_echleon as r
from tqdm import tqdm
from cProfile import Profile
from sklearn.preprocessing import normalize
from pstats import Stats
import re
import sys
from data_structures import VariableNode, CheckNode, ValueTree, Link

"""
class Node:

    def __init__(self, no_connections, identifier, value=0):
        self.value = value
        self.links = np.zeros(no_connections, dtype=int)
        self.identifier = identifier

    def add_link(self, node):
        Adds a link to the node. Throws an error if the node is full 
        for (i,j) in enumerate(self.links):
            if not j:
                self.links[i] = node
                break
        
        return self.links
    
    def get_links(self):
        return self.links

    def replace_link(self, node, index):
        Replaces a link with another link 
        self.links[index] = node
        return self.links
    
    def change_value(self, new_value):
        self.value = new_value
    
class CheckNode(Node):

    def __init__(self, dc, identifier):
        super().__init__(dc, identifier)
    
class VariableNode(Node):
    def __init__(self, dv, identifier, value=0):
        super().__init__(dv, identifier, value)

    
class Link(Node):
    def __init__(self, cn, vn, value):
        self.cn = cn
        self.vn = vn
        self.value = value
"""

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

    pdf = conv_circ(arr_pd[0], arr_pd[1])

    for i in arr_pd[2:]:
        pdf = conv_circ(pdf, i)

    return pdf

class VariableTannerGraph:
    """ Initializes empty, on establishing connections creates H and forms links """

    def __init__(self, dv, dc, k, n, ffdim=2):
        
        # Check if connections are non-uniform
        if type(dv) == list:
            assert len(dv) == n and len(dc) == n-k
            self.vns = [VariableNode(dv[i], i) for i in range(n)]
            self.cns = [CheckNode(dc[i], i) for i in range(n-k)]
            self.dv = dv
            self.dc = dc
        else:
            self.vns = [VariableNode(dv, i) for i in range(n)]
            self.cns = [CheckNode(dc, i) for i in range(n-k)]
            self.dv = [dv for i in range(n)]
            self.dc = [dc for i in range(n-k)]

        # For the singular case - it remains as an integer, but for the Changing Case it goes to a list, need to make sure that does not break everything
        self.k = k
        self.n = n
        self.ffdim = ffdim
        self.links = {}
        self.vn_values = [i.value for i in self.vns]

    def add_link(self, cn_index, vn_index, link_value):
        """ Adds a link to the links data structure """
        self.links[(cn_index, vn_index)] = link_value
    
    def update_link_weight(self, cn_index, vn_index, link_value):
        """ Updates Link weight """
        self.add_link(cn_index, vn_index, link_value)
    
    def get_link_weight(self, cn_index, vn_index):
        """ Get Link Weight """
        return self.links[(cn_index, vn_index)]
    
    def update_within_link_weight(self, cn_index, vn_index, val_index, new_value):
        self.links[(cn_index, vn_index)][val_index] = new_value

    def get_vn_value(self, vn_index):
        return self.vns[vn_index].get_value()

    def get_cn_value(self, cn_index):
        return self.cns[cn_index].get_value()

    def establish_connections(self, Harr=None):
        """ Establishes connections between variable nodes and check nodes """
        
        # In case Harr is sent as a parameter
        if Harr is None:
            # If we are creating, assuming it's not scldpc - really needs some unification here champ
            self.Harr = r.get_H_arr(self.dv[0], self.dc[0], self.k, self.n)
        else:
            self.Harr = np.array(Harr)

        # Our Harr is implementation is different - will need to be considered when adapting - assuming that this is the check nodes they are connected to
        Harr = self.Harr

        # Divide Harr into dv parts  
        # But dv is a list in the case of the changing case
        # All the dvs are the same for this case
        dv = self.dv[0]

        if len(np.unique(self.dc)) == 1:
            Harr = Harr // self.dc[0]
        
        Harr = [Harr[i:i+dv] for i in range(0, len(Harr), dv)]

        # Checking for spatially coupled
        
        
        # Establish connections
        for (i,j) in enumerate(Harr):
            for k in j:
                self.vns[i].add_link(self.cns[k])
                self.cns[k].add_link(self.vns[i])

    def get_connections(self):
        """ Returns the connections in the Tanner Graph """
        return [(i.get_value(), j) for i in self.cns for j in i.links]

    def get_cn_link_values(self, cn):
        """ Returns the values of the link weights for the cn as an array"""
        vals = []
        for i in cn.links:
            vals.append(self.get_link_weight(cn.identifier, i))

        return vals
    
    def get_total_possibilities(self):
        """ Returns the total number of possbilities in all the cn's - for the CC Decoder"""
        return sum([i.get_total_symbol_possibilities() for i in self.cns])
    
    def get_no_unresolved_vns(self):
        """ Returns the total number of unresolved vns (CC Decoder)"""
        return len([i for i in self.vns if i.get_total_symbol_possibilities() > 1])
    
    def assign_values(self, arr):   
        """Assigns values to the VNs based on input pre decoding """

        assert len(arr) == len(self.vns) 

        for i in range(len(arr)):
            self.vns[i].change_value(arr[i])

    def get_max_prob_codeword(self, P, GF):
        """Calculates the most possible Codeword using the probability likelihoods established in the VN's and influenced by the initial probability likelihoods.

        Returns:
            codeword (arr): n length most probable codeword with symbols
        """
        z = np.zeros(self.n)
        for j in self.vns:
            vn_index = j.identifier
            probs = 1 * P[vn_index]
            for a in range(GF.order):
                for i in j.links:
                    probs[a] *= self.get_link_weight(i, vn_index)[a]
            z[vn_index] = np.argmax(probs) 
        z = GF(z.astype(int))
        return z

    def normalize(self, arr):
        """ Normalizes an array """
        sum_arr = sum(arr)
        return [i/sum_arr for i in arr]

    def validate_codeword(self, H, GF, max_prob_codeword):
        """ Checks if the most probable codeword is valid as a termination condition of qspa decoding """
        return not np.matmul(H, max_prob_codeword).any()

    def remove_from_array(self, vals, current_value):
        """ Removes current value from vals"""

        new_vals = []
        for i in range(len(vals)):
            if np.array_equal(vals[i], current_value):
                continue
            new_vals.append(vals[i])
        return new_vals 

    def initialize_vn_links(self, P):
        """ Sets all the links from a VN to the VN initial likelihood array """
        for i in self.vns:
            vn_index = i.identifier
            for j in i.links:
                self.update_link_weight(j, vn_index, 1*P[vn_index])

    def update_cn_links(self, cn, new_vals):
        """ Updates the CN links post a VN update iteration """
        cn_index = cn.identifier
        for i,j in enumerate(cn.links):
            self.update_link_weight(cn_index, j, new_vals[i])

    def vn_update_qspa(self):
        """ VN Update for the QSPA Decoder. For each CN, performs convolutions for individual VN's as per the remaining links and updates the individual link values after finishing each link. Repeats for all the CN's """
        
        for i in self.cns:
            cn_index = i.identifier
            vns = i.links
            new_pdfs = []
            for j in vns:
                conv_indices = vns[vns!=j]
                vals = [self.get_link_weight(cn_index, t) for t in conv_indices]
                pdf = perform_convolutions(vals)
                new_pdfs.append(pdf[self.idx_shuffle])
                #self.update_link_weight(i,j,pdf[idx_shuffle]) 
            self.update_cn_links(i, new_pdfs)
            

    def cn_update_qspa(self):
        """ Updates the CN as per the QSPA Decoding. Conditional Probability of a Symbol being favoured yadayada """

        copy_links = self.links.copy()
        for a in range(self.GF.order):
            for j in self.vns:
                vn_index = j.identifier
                for i in j.links:
                    copy_links[(i, vn_index)][a] = self.P[vn_index][a]
                    for t in j.links[j.links!=i]:
                        copy_links[(i,vn_index)][a] *= self.get_link_weight(t, vn_index)[a]

                    copy_links[i, vn_index] = self.normalize(copy_links[(i,vn_index)])    
        self.links = copy_links

    def qspa_decoding(self, H, GF, max_iterations=10):

        self.GF = GF
              
        # Additive inverse of GF Field
        self.idx_shuffle = np.array([
            (GF.order - a) % GF.order for a in range(GF.order)
        ])
        
        # Initial likelihoods
        self.P = [i.value for i in self.vns]

        self.initialize_vn_links(self.P)
        
        copy_links = self.links.copy()
        max_prob_codeword = self.get_max_prob_codeword(self.P, GF)

        for i in range(max_iterations):
            
            self.vn_update_qspa()

            max_prob_codeword = self.get_max_prob_codeword(self.P, GF)
            if self.validate_codeword(H, GF, max_prob_codeword):
                print("Decoding converges")
                return max_prob_codeword

            self.cn_update_qspa()
        
        print("Decoding does not converge")
        return max_prob_codeword
    
    def coupon_collector_decoding(self, max_iterations=10000):
        """ Decodes for the case of symbol possiblities for each variable node 
            utilising Belief Propagation - may be worth doing for BEC as well 
        """
        
        unresolved_vns = self.get_no_unresolved_vns()
        iterations = 0
        resolved_vns = 0
        total_possibilites = self.get_total_possibilities()
        print(total_possibilites)
        decoded_values = [i.get_value() for i in self.vns]
        
        if total_possibilites >= 67*len(self.vns):
            return np.random.rand(2, len(self.vns))

        while True:
            # Iterating through all the check nodes
            for i in self.cns:
                
                vn_vals = [j.get_value() for j in i.links]
                
                for j in i.links:
                
                    vals = vn_vals.copy()
                    current_value = j.get_value()
                    vals.remove(current_value)

                    possibilites = permuter(vals, self.ffdim, current_value)
                    new_values = list(set(current_value).intersection(set(possibilites)))
                    j.change_value(new_values)

                    """
                    if len(new_values) < len(current_value) and len(possibilites) > 1:
                        print("I reached here")
                    """
                    if len(current_value) > 1 and len(new_values) == 1:
                        resolved_vns += 1
                        decoded_values[j.identifier] = new_values
                    
                if unresolved_vns ==  resolved_vns and sum([len(i) for i in decoded_values]) == len(decoded_values):
                    return np.array([i.get_value() for i in self.vns])
            
            print(self.get_total_possibilities())
            if iterations > 3:
                print("Puedo entrar")
                print(unresolved_vns - resolved_vns)
                return [i.get_value() for i in self.vns]
            """
            if self.get_total_possibilities() == total_possibilites:
                print("Puedo entrar")
                return [i.get_value() for i in self.vns]
            """
            iterations += 1
            
            total_possibilites = self.get_total_possibilities()
            
        return [i.value for i in self.vns]
    
    def adaptive_coupon_collector_decoding(self, max_iterations=10000):
        """ Decodes for the case of symbol possiblities for each variable node 
            utilising Belief Propagation - may be worth doing for BEC as well 
        """
        
        unresolved_vns = self.get_no_unresolved_vns()
        resolved_vns = 0
        total_possibilites = self.get_total_possibilities()
        decoded_values = [vn.get_value() for vn in self.vns]
        
        # Adding condition - in case we know for sure we cannot decode
        if total_possibilites >= 67*len(self.vns):
            #print("I Enter here")
            return np.random.rand(2, len(self.vns))

        iterations = 0
        #print("Created Value Tree")

        while True:
            
            tree = ValueTree(self.cns)
            #print(iterations)
            #for i in range(len(self.cns)):
            while not tree.is_empty():
                #print("I enter here")   
                # Get smallest node (check node with least possibilities)
                cn = tree.remove_smallest_node()
                #print(cn.get_total_symbol_possibilites())
                vn_vals = [vn.get_value() for vn in cn.links]
                    
                for vn in cn.links:
                    vals = vn_vals.copy()
                    current_value = vn.get_value()
                    vals.remove(current_value)

                    possibilites = permuter(vals, self.ffdim, current_value)
                    new_values = list(set(current_value).intersection(set(possibilites)))
                    vn.change_value(new_values)
                    
                    if len(current_value) > 1 and len(new_values) == 1:
                        resolved_vns += 1
                        decoded_values[vn.identifier] = new_values

                # Adding node back into the tree        
                #tree.add_node(cn)

            if unresolved_vns==resolved_vns: #and sum([len(i) for i in decoded_values]) == len(decoded_values):
                return np.array([i.value for i in self.vns])
        
            # No certainty gained after a whole iteration
            if self.get_total_possibilities() == total_possibilites:
                return [i.value for i in self.vns]
            
            total_possibilites = self.get_total_possibilities()
            
            iterations+=1
    
        return [i.value for i in self.vns]