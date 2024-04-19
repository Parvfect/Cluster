
from qspa_conv import hw_likelihoods, QSPADecoder
import utils
import galois
import os
import row_echleon as r
import numpy as np
from itertools import combinations
from tqdm import tqdm
from protograph_interface import get_Harr_sc_ldpc, get_dv_dc
from tanner_original import VariableTannerGraph
import random
import matplotlib.pyplot as plt
from pstats import Stats
import re
from coupon_collector import get_parameters, get_parameters_sc_ldpc
from tanner_qspa import TannerQSPA
from coupon_collector import generate_run_save_file

def choose_symbols(n_motifs, picks):
    """Creates Symbol Array as a combination of Motifs
    
    Args: 
        n_motifs (int): Total Number of Motifs
        picks (int): Number of Motifs per Symbol
    Returns: 
        symbols (list): List of all the Symbols as motif combinations
    """

    # Reference Motif Address starts from 1 not 0
    return [list(i) for i in (combinations(np.arange(1, n_motifs+1), picks))]

def distracted_coupon_collector_channel(symbol, R, P, n_motifs):
    """Model of the Distracted Coupon Collector Channel. Flips a coin, if the probability is within interference, randomly attach a motif from the set of all motifs. Otherwise randomly select from the set of motifs for the passed symbol
    
    Args: 
        symbol (list) : List of motifs as a symbol
        R (int): Read Length
        P (float): Probability of Interference 
        n_motifs (int): Number of motifs in Total
    
    Returns: 
        reads (list) : List of Reads for the Symbol
    """

    reads = []
    for i in range(R):
        if random.random() > P:
            reads.append(random.choice(symbol))
        else:
            reads.append(random.randint(1, n_motifs))    
    return reads

def get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim, zero_codeword=False, display=True, Harr=None, H=None, G=None):
    """ Returns the parameters for the simulation """

    # Starting adresses from 1
    motifs = np.arange(1, n_motifs+1)
    
    symbols = choose_symbols(n_motifs, n_picks)
    
    symbols.pop(-1)
    symbols.pop(-2)
    symbols.pop(-3)
    
    symbol_keys = np.arange(0, ffdim)

    #graph = VariableTannerGraph(dv, dc, k, n, ffdim=ffdim)
    graph = TannerQSPA(dv, dc, k, n, ffdim=ffdim)

    if Harr is None:
        Harr = r.get_H_arr(dv, dc, k, n)
        H = r.get_H_Matrix(dv, dc, k, n, Harr)
        if zero_codeword:
            G = np.zeros([k,n], dtype=int)
        else:
            G = r.parity_to_generator(H, ffdim=ffdim)

    graph.establish_connections(Harr)
    
    if np.any(np.dot(G, H.T) % ffdim != 0):
        print("Matrices are not valid, aborting simulation")
        exit()

    input_arr = [random.choice(symbol_keys) for i in range(k)]

    # Encode the input array
    C = np.dot(input_arr, G) % ffdim

    # Check if codeword is valid
    if np.any(np.dot(C, H.T) % ffdim != 0):
        print("Codeword is not valid, aborting simulation")
        exit()

    return H, G, graph, C, symbols, motifs

def get_parameters_sc_ldpc(n_motifs, n_picks, L, M, dv, dc, k, n, ffdim, zero_codeword=False, display=True, Harr=None, H=None, G=None):
    """ Returns the parameters for the simulation """

    # Starting adresses from 1
    motifs = np.arange(1, n_motifs+1)
    
    symbols = choose_symbols(n_motifs, n_picks)
    
    symbols.pop(-1)
    symbols.pop(-2)
    symbols.pop(-3)
    
    symbol_keys = np.arange(0, ffdim)
    
    if Harr is None:
        Harr, dv, dc, k, n = get_Harr_sc_ldpc(L, M, dv, dc)   
    else:
        dv, dc = get_dv_dc(dv, dc, k, n, Harr)
    
    #graph = VariableTannerGraph(dv, dc, k, n, ffdim=ffdim)
    graph = TannerQSPA(dv, dc, k, n, ffdim=ffdim)
    graph.establish_connections(Harr)

    if H is None and G is None:
        H = r.get_H_matrix_sclpdc(dv, dc, k, n, Harr)
        if zero_codeword:
            G = np.zeros([k,n], dtype=int)
        else:
            G = r.parity_to_generator(H, ffdim=ffdim)

    if np.any(np.dot(G, H.T) % ffdim != 0):
        print("Matrices are not valid, aborting simulation")
        exit()

    input_arr = [random.choice(symbol_keys) for i in range(k)]
    
    C = np.dot(input_arr, G) % ffdim

    # Check if codeword is valid
    if np.any(np.dot(C, H.T) % ffdim != 0):
        print("Codeword is not valid, aborting simulation")
        exit()

    return H, G, graph, C, symbols, motifs


def get_symbol_likelihood(n_picks, motif_occurences, P, pop=True):
    """Generates Likelihood Array for a Symbol after it's passed through the channel, using the number of times each motif is encountered

        Args:
            n_picks (int): Number of motifs per symbol
            motif_occurences (array) (n_motifs,): Array of Occurence of Each Motif Encountered [0,0,1,1,2,3,0] 
            P (float): Interference Probability
        Returns:
            likelihoods: array (n_motifs choose k_motifs, ) - Normalized likelihood for each symbol (in lexicographical order).
    """

    # Getting the Likelihoods from Alberto's Likelihood Generator
    likelihoods = hw_likelihoods(n_picks, motif_occurences, P)

    if pop:
        # Popping the last three likelihoods to make the symbols match
        likelihoods.pop()
        likelihoods.pop()
        likelihoods.pop()

    if sum(likelihoods) == 0: # Prevent divide by zero
        likelihoods = list(np.ones(67)/67)
    else:
        norm_factor = 1/sum(likelihoods)
        likelihoods = [norm_factor*i for i in likelihoods]
        
    # Precision - summing up to 0.99..
    assert sum(likelihoods) >= 0.99 and sum(likelihoods) < 1.01

    return likelihoods

def simulate_reads(C, symbols, read_length, P, n_motifs, n_picks):
    """Simulates reads using the QSPA Decoder
        Args:
            C (list) (n,): Codeword
            read_length (int): Read Length
            P (Float): Interference Probability
            n_motifs (int): Number of Motifs in Total
            n_picks (int): Number of Motifs Per Symbol
        Returns: 
            reads (list) : [length of Codeword, no. of symbols] list of all the reads as likelihoods
    """

    likelihood_arr = []
    for i in C:
        motif_occurences = np.zeros(n_motifs)
        reads = distracted_coupon_collector_channel(symbols[i], read_length, P, n_motifs)

        # Collecting Motifs Encountered
        for i in reads:
            motif_occurences[i-1] += 1

        symbol_likelihoods = get_symbol_likelihood(n_picks, motif_occurences, P)
        likelihood_arr.append(symbol_likelihoods)

    return likelihood_arr

def decoding_errors_fer(k, n, dv, dc, P, H, G, GF, graph, C, symbols, n_motifs, n_picks, decoder=None, decoding_failures_parameter=10, max_iterations=10, iterations=50, uncoded=False, bec_decoder=False, label=None, code_class="", read_lengths=np.arange(1,20), plot=True):

    frame_error_rate = []
    max_iterations = max_iterations
    decoding_failures_parameter = decoding_failures_parameter # But can be adjusted as a parameter

    for i in tqdm(read_lengths):
        decoding_failures, iterations, counter = 0, 0, 0
        for j in tqdm(range(max_iterations)):
            symbol_likelihoods_arr = np.array(simulate_reads(C, symbols, i, P, n_motifs, n_picks))

            if not decoder:
                z = graph.qspa_decode(symbol_likelihoods_arr, H, GF)
            else:
                z = decoder.decode(symbol_likelihoods_arr, max_iter=20)
            
            #print(C)
            #print(z)

            if np.array_equal(C, z):
                counter += 1
            else: 
                decoding_failures+=1
            
            iterations += 1
            if decoding_failures == decoding_failures_parameter:
                break
            
            
        assert counter == (iterations - decoding_failures)
        error_rate = (iterations - counter)/iterations
        frame_error_rate.append(error_rate)

    write_path = os.path.join(os.environ['HOME'], "results.txt")
    with open(write_path, "a") as f:
        f.write(f"These are the results \n {frame_error_rate}")

    if plot:
        plt.plot(read_lengths, frame_error_rate, 'o')
        plt.plot(read_lengths, frame_error_rate, label=label)
        plt.title("Frame Error Rate for DCC for {}{}-{}  {}-{} for 8C4 Symbols".format(code_class, k, n, dv, dc))
        plt.ylabel("Frame Error Rate")
        plt.xlabel("Read Length")

        # Displaying final figure
        plt.xlim(read_lengths[0], read_lengths[-1])
        plt.ylim(0,1)
        plt.xticks(np.arange(read_lengths[0], read_lengths[-1], 1))

    return frame_error_rate

def decoding_errors(k, n, dv, dc, P, H, G, GF, graph, C, symbols, n_motifs, n_picks, decoder=None, decoding_failures_parameter=10, max_iterations=500, iterations=50, uncoded=False, bec_decoder=False, label=None, code_class="", read_length=14, plot=True):

    frame_error_rate = []
    max_iterations = max_iterations
    decoding_failures_parameter = decoding_failures_parameter # But can be adjusted as a parameter

    iterations = 0
    decoding_failures = 0

    for j in tqdm(range(max_iterations)):
        symbol_likelihoods_arr = np.array(simulate_reads(C, symbols, read_length, P, n_motifs, n_picks))

        z = graph.qspa_decode(symbol_likelihoods_arr, H, GF)
        
        if not np.array_equal(C, z):
            decoding_failures+=1
        
        iterations += 1
    
    print(f"\nIterations {iterations} Failures {decoding_failures}")

    write_path = os.path.join(os.environ['HOME'], "results2.txt")
    with open(write_path, "a") as f:
        f.write(f"\nIterations {iterations} Failures {decoding_failures}")


def run_fer(n_motifs, n_picks, dv, dc, k, n, L, M, ffdim, P, code_class="", iterations=10, cc_decoder=False, bec_decoder=False, uncoded=False, read_lengths=np.arange(1,20), max_iter=10,  zero_codeword=False, graph_decoding=False, label=None):
    
    if code_class == "sc_":
        H, G, graph, C, symbols, motifs = get_parameters_sc_ldpc(n_motifs, n_picks, L, M, dv, dc, k, n, ffdim, zero_codeword=zero_codeword, display=False, Harr=None, H=None, G=None)
    else:
        H, G, graph, C, symbols, motifs = get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim, zero_codeword=zero_codeword, display=False, Harr=None, H=None, G=None)
    
    GF = galois.GF(ffdim)
    GFH = GF(H.astype(int)) # * GF(np.random.choice(GF.elements[1:], siz
    GFK = GF(G.astype(int))


    decoding_errors(k, n, dv, dc, P, GFH, GFK, GF, graph, C, symbols, n_motifs, n_picks, label="Graph QSPA", read_length=read_lengths[0])
    #plt.legend()
    #plt.grid()
    #plt.show()
    

if __name__ == "__main__":
    n_motifs, n_picks = 8, 4

    # dv - Variable Node Connections
    # dc - Check Node Connections
    # ffdim - Finite field Dimensions
    # P - Interference Probability
    dv, dc, ffdim, P = 3, 9, 67, 2 * 0.038860387943791645
    # n = Number of Variable Nodes, k = Number of Variable Nodes - Number of Check Nodes 
    k, n = 200, 300
    # SC LDPC Protograph Parameters - L, M
    L, M = 15, 51
    read_length = 6
    read_lengths = np.arange(10, 11)


    #run_fer(n_motifs, n_picks, dv, dc, k, n, L, M, ffdim, P, code_class="",  uncoded=False, zero_codeword=False, bec_decoder=False, graph_decoding=False, read_lengths=read_lengths)
    run_fer(n_motifs, n_picks, dv, dc, k, n, L, M, ffdim, P, code_class="sc_",  uncoded=False, zero_codeword=True, bec_decoder=False, graph_decoding=True,  read_lengths=read_lengths)
                                                                                                    

