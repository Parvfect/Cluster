
from tanner import VariableTannerGraph, conv_circ
import numpy as np
import random

def get_max_symbol(prob_arr):
    max_val = np.max(prob_arr)

    # Numerical issues ? Strict Equality?
    max_indices = [i for i, val in enumerate(prob_arr) if val == max_val]
    #print(prob_arr)
    #print(max_indices)
    return random.choice(max_indices)

class TannerQSPA(VariableTannerGraph):

    def __init__(self, dv, dc, k, n, ffdim=2):
        super().__init__(dv,dc,k,n,ffdim)

        self.vn_links = {}
        self.cn_links = {}

    def initialize_vn_links(self, P):
        """ Sets all the links from a VN to the VN initial likelihood array """
        for vn in self.vns:
            vn_index = vn.identifier
            for cn_index in vn.links:
                self.vn_links[(cn_index, vn_index)] = P[vn_index].copy()

    def initialize_cn_links(self):
        """ Initializes CN Links"""
        for cn in self.cns:
            cn_index = cn.identifier
            for vn_index in cn.links:
                self.cn_links[(cn_index, vn_index)] = np.zeros(67)
        
    def qspa_decode(self, symbol_likelihood_arr, H, GF, max_iterations=10):
        """Decodes using QSPA """

        self.GF = GF
                
        # Additive inverse of GF Field
        self.idx_shuffle = np.array([
            (GF.order - a) % GF.order for a in range(GF.order)
        ])
        
        # Setting the VN Links with the initial symbol likelihoods
        self.initialize_vn_links(symbol_likelihood_arr)

        # Initilizing the CN Links
        self.initialize_cn_links()

        prev_max_prob_codeword = self.get_max_prob_codeword(symbol_likelihood_arr, GF)

        iterations = 0

        #for i in range(max_iterations):
        while(True):
            
            
            self.cn_update()

            #print()
            #print(f"Iteration {iterations+1}")
            #print()
            max_prob_codeword = self.get_max_prob_codeword(symbol_likelihood_arr, GF)
            #print(max_prob_codeword)

            #print(sum(random.choice(list(self.cn_links.items()))[1]))

            parity = not np.matmul(H, max_prob_codeword).any()
            
            
            if parity:
                #print("Decoding converges")
                return max_prob_codeword
                

            self.vn_update(symbol_likelihood_arr)
            #print(sum(random.choice(list(self.vn_links.items()))[1]))

            
            if np.array_equal(max_prob_codeword, prev_max_prob_codeword) or iterations > max_iterations:
                break
            
            prev_max_prob_codeword = max_prob_codeword

            iterations+=1
            #print(f"Iteration {iterations}")

        #print("Decoding does not converge")
        return max_prob_codeword
    
    def get_max_prob_codeword(self, P, GF):
        """Calculates the most possible Codeword using the probability likelihoods established in the VN's and influenced by the initial probability likelihoods.

        Returns:
            codeword (arr): n length most probable codeword with symbols
        """
        
        # Initialize Empty Array
        z = np.zeros(self.n)
        
        # Iterate Through all the VNs
        for vn in self.vns:
        
            vn_index = vn.identifier
            probs = 1 * P[vn_index]
            
            # Iterate Through Each Symbol Possibility
            for a in range(GF.order):
                
                # Iterate Through all the CNs connected
                for cn in vn.links:

                    # Update Symbol Probability as product of the CN Message
                    probs[a] *= self.cn_links[(cn, vn_index)][a]
            
            # Most likely symbol is the Symbol with the highest probability
            #print(probs)
            z[vn_index] = get_max_symbol(probs)
            #print(z)
        
        return GF(z.astype(int))

    

    def cn_update(self):
        """ CN Update for the QSPA Decoder. For each CN, performs convolutions for individual VN's as per the remaining links and updates the individual link values after finishing each link. Repeats for all the CN's """
        
        # Iterate through all the CNs
        for cn in self.cns:

            cn_index = cn.identifier

            vns = cn.links
            # Iterating through all the VN Links of the Check node
            for vn in vns:

                # Getting all the remaining VNS
                conv_indices = [idx for idx in vns if idx != vn]

                # Getting convolution of all the vns
                pdf = conv_circ(self.vn_links[(cn_index, conv_indices[0])], self.vn_links[(cn_index, conv_indices[1])])

                for indice in conv_indices[2:]:
                    pdf = conv_circ(pdf, self.vn_links[(cn_index, indice)])

                # Updating the CN Link weight with the conv value
                self.cn_links[(cn_index, vn)] = pdf[self.idx_shuffle]
                #print(sum(self.cn_links[(cn_index, vn)]))

    def vn_update(self, P):
        """ Updates the CN as per the QSPA Decoding. Conditional Probability of a Symbol being favoured yadayada """

        # Use the CN links to update the VN links by taking the favoured probabilities
        
        # Iterating through all the Symbols
        
        # For each VN
        for vn in self.vns:
            vn_index = vn.identifier
            
            for cn in vn.links:
                for a in range(self.GF.order):
                
                    self.vn_links[(cn, vn_index)][a] = P[vn_index][a]

                    for t in vn.links:
                        # Iterating through all the other cns besides selected
                        if t == cn:
                            continue

                        self.vn_links[(cn, vn_index)][a] *= self.cn_links[(t, vn_index)][a]

            
                # Normalizing
                #sum_copy_links = np.einsum('i->', self.vn_links[(cn, vn_index)])
                sum_copy_links = np.sum(self.vn_links[(cn, vn_index)])
                self.vn_links[(cn, vn_index)] = self.vn_links[(cn, vn_index)]/sum_copy_links


                    


