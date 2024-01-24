
import random
import numpy as np
import os
import string
from graph import TannerGraph
from tanner import VariableTannerGraph
import row_echleon as r
from scipy.linalg import null_space
import sympy as sympy
from itertools import combinations
from pstats import Stats
import re
from cProfile import Profile
from tqdm import tqdm
import matplotlib.pyplot as plt
from protograph_interface import get_Harr_sc_ldpc, get_dv_dc
import sys


def choose_symbols(n_motifs, picks):
    """ Returns Symbol Dictionary given the motifs and the number of picks """
    return [list(i) for i in (combinations(np.arange(1, n_motifs+1), picks))]

def coupon_collector_channel(symbol, R, visibility=1):
    reads = []
    for i in range(R):
        if random.random() < visibility:
            reads.append(random.choice(symbol))
    return reads

def get_symbol_index(symbols, symbol):
    for i in symbols:
        if set(i) == set(symbol):
            return symbols.index(i)

def get_possible_symbols(reads, symbols, motifs, n_picks):
    
    reads = [set(i) for i in reads]
    symbol_possibilities = []
    for i in reads:

        motifs_encountered = i
        motifs_not_encountered = set(motifs) - set(motifs_encountered)
        read_symbol_possibilities = []

        # For the case of distraction
        if len(motifs_encountered) > n_picks:
            return symbols

        if len(motifs_encountered) == n_picks:
            read_symbol_possibilities = [get_symbol_index(symbols, motifs_encountered)]
        
        else:
            remaining_motif_combinations = [set(i) for i in combinations(motifs_not_encountered, n_picks - len(motifs_encountered))]
            
            for i in remaining_motif_combinations:
                possible_motifs = motifs_encountered.union(i)
                symbols = [set(j) for j in symbols]
                if possible_motifs in symbols:
                    read_symbol_possibilities.append(get_symbol_index(symbols, motifs_encountered.union(i)))
        symbol_possibilities.append(read_symbol_possibilities)
    
    return symbol_possibilities
 
def simulate_reads(C, read_length, symbols):
    """ Simulates the reads from the coupon collector channel """
    
    reads = []
    for i in C:
        read = coupon_collector_channel(symbols[i], read_length)
        reads.append(read)
    return reads

def read_symbols(C, read_length, symbols, motifs, picks):
    reads = simulate_reads(C, read_length, symbols)
    return get_possible_symbols(reads, symbols, motifs, picks)

def get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim, display=True, Harr=None, H=None, G=None):
    """ Returns the parameters for the simulation """

    # Starting adresses from 1
    motifs = np.arange(1, n_motifs+1)
    
    symbols = choose_symbols(n_motifs, n_picks)
    
    symbols.pop(-1)
    symbols.pop(-2)
    symbols.pop(-3)
    
    symbol_keys = np.arange(0, ffdim)

    graph = TannerGraph(dv, dc, k, n, ffdim=ffdim)

    if Harr is None:
        Harr = r.get_H_arr(dv, dc, k, n)
        H = r.get_H_Matrix(dv, dc, k, n, Harr)
        #G = r.parity_to_generator(H, ffdim=ffdim)
        G = r.alternative_parity_to_generator(H, ffdim=ffdim)

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

    return Harr, H, G, graph, C, symbols, motifs

def get_parameters_sc_ldpc(n_motifs, n_picks, L, M, dv, dc, k, n, ffdim, display=True, Harr=None, H=None, G=None):
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
    
    graph = VariableTannerGraph(dv, dc, k, n, ffdim=ffdim)
    graph.establish_connections(Harr)

    if H is None and G is None:
        H = r.get_H_matrix_sclpdc(dv, dc, k, n, Harr)
        
        print(np.linalg.matrix_rank(H))
        print(H.shape)
        #print(H.rank)
        G = r.parity_to_generator(H, ffdim=ffdim)
        print(G.shape)
        G1 = r.alternative_parity_to_generator(H, ffdim=ffdim)
        print(G1.shape)

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

    return Harr, H, G, graph, C, symbols, motifs, k, n


def decoding_errors_fer(k, n, dv, dc, graph, C, symbols, motifs, n_picks, read_lengths = np.arange(1,12), decoding_failures_parameter=5, max_iterations=1000, iterations=50, uncoded=False, bec_decode=False, label=None, code_class=""):
    """ Returns the frame error rate curve - for same H, same G, same C"""

    frame_error_rate = []
    max_iterations = max_iterations
    decoding_failures_parameter = decoding_failures_parameter # But can be adjusted as a parameter

    for i in tqdm(read_lengths):
        decoding_failures, iterations, counter = 0, 0, 0
        for j in tqdm(range(max_iterations)):
            # Assigning values to Variable Nodes after generating erasures in zero array
            symbols_read = read_symbols(C, i, symbols, motifs, n_picks)
            if not uncoded:
                graph.assign_values(read_symbols(C, i, symbols, motifs, n_picks))
                if bec_decode:
                    decoded_values = graph.coupon_collector_erasure_decoder()
                else:
                    decoded_values = graph.coupon_collector_decoding()
            else:
                decoded_values = symbols_read
            
            if sum([len(i) for i in decoded_values]) == len(decoded_values): # Checks if we have decoded completely
                if np.all(np.array(decoded_values).T[0] == C):
                    counter += 1    
            else: 
                decoding_failures+=1

            iterations += 1
            
            if decoding_failures == decoding_failures_parameter:
                break

        assert counter == (iterations - decoding_failures)
        error_rate = (iterations - counter)/iterations
        frame_error_rate.append(error_rate)
    
    plt.plot(read_lengths, frame_error_rate, 'o')
    plt.plot(read_lengths, frame_error_rate, label=label)
    plt.title("Frame Error Rate for CC for {}{}-{}  {}-{} for 8C4 Symbols".format(code_class, k, n, dv, dc))
    plt.ylabel("Frame Error Rate")
    plt.xlabel("Read Length")

    # Displaying final figure
    plt.xlim(read_lengths[0], read_lengths[-1])
    plt.ylim(0,1)

    return frame_error_rate

def run_fer(n_motifs, n_picks, dv, dc, k, n, L, M, ffdim, read_lengths=np.arange(1,12), code_class="", iterations=5, bec_decoder=False, uncoded=False, saved_code=False, singular_decoding=True, fer_errors=True):

    Harr, H, G = None, None, None

    #if saved_code:
    #    Harr, H, G = get_saved_code(dv, dc, k, n, L, M, code_class=code_class)
    
    if code_class == "sc_":
        Harr, H, G, graph, C, symbols, motifs, k, n = get_parameters_sc_ldpc(n_motifs, n_picks, L, M, dv, dc, k, n, ffdim, display=False, Harr=Harr, H=H, G=G)
    else:
        Harr, H, G, graph, C, symbols, motifs = get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim, display=False, Harr =Harr, H=H, G=G)
    
    fer = decoding_errors_fer(k, n, dv, dc, graph, C, symbols, motifs, n_picks, read_lengths=read_lengths, iterations=iterations, label=f'CC Decoder', code_class=code_class)
    label = 'Coupon Collector'
    
    if bec_decoder:
        fer = decoding_errors_fer(k, n, dv, dc, graph, C, symbols, motifs, n_picks, read_lengths=read_lengths, iterations=iterations, bec_decode=True, label=f'BEC Decoder', code_class=code_class)
        label = "BEC"
    
    if uncoded:
        fer = decoding_errors_fer(k, n, dv, dc, graph, C, symbols, motifs, n_picks, read_lengths=read_lengths, iterations=iterations, uncoded=True, label=f'Uncoded', code_class=code_class)
        label = 'Uncoded'
    
    generate_run_save_file(n_motifs, n_picks, dv,dc, k, n, L, M, motifs, symbols, Harr, H, G, C, ffdim, code_class, fer, read_lengths, label)

def generate_run_save_file(n_motifs, n_picks, dv, dc, k, n, L, M, motifs, symbols, Harr, H, G, C, ffdim, code_class, fer, read_lengths, label):
    uid = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(10))
    os.mkdir(f"Run-code-dv-dc-k-n-label={code_class}-{dv}-{dc}-{k}-{n}-{label}{uid}")
    np.save(f"Run-code-dv-dc-k-n-label={code_class}-{dv}-{dc}-{k}-{n}-{label}{uid}/Harr.npy", Harr)
    np.save(f"Run-code-dv-dc-k-n-label={code_class}-{dv}-{dc}-{k}-{n}-{label}{uid}/H.npy", H)
    np.save(f"Run-code-dv-dc-k-n-label={code_class}-{dv}-{dc}-{k}-{n}-{label}{uid}/G.npy", G)
    np.save(f"Run-code-dv-dc-k-n-label={code_class}-{dv}-{dc}-{k}-{n}-{label}{uid}/C.npy", C)
    with open(f"Run-code-dv-dc-k-n-label={code_class}-{dv}-{dc}-{k}-{n}-{label}{uid}/savefile.txt", 'w') as f:
        f.write("The number of motifs are {}\n".format(n_motifs))
        f.write("The number of picks are {}\n".format(n_picks))
        f.write("The dv is {}\n".format(dv))
        f.write("The dc is {}\n".format(dc))
        f.write("The k is {}\n".format(k))
        f.write("The n is {}\n".format(n))
        f.write(f"The L is {L}\n")
        f.write(f"The M is {M}\n")
        f.write("GF{}\n".format(ffdim))
        f.write("The Motifs are \n{}\n".format(motifs))
        f.write("The Symbols are \n{}\n".format(symbols))
        f.write("The Harr is in \n{}\n".format("Harr.npy"))
        f.write("The Parity Matrice is in \n{}\n".format("H.npy"))
        f.write("The Generator Matrix is in \n{}\n".format("G.npy"))
        f.write("The Codeword is in \n{}\n".format("C.npy"))

        f.write(f"The Read Lengths are \n{read_lengths}\n")
        f.write(f"The Frame Error rate is \n{fer}\n")

    print(fer)

    plt.plot(read_lengths, fer)
    plt.xticks(np.arange(read_lengths[0], read_lengths[-1], 1))
    plt.grid()
    plt.legend()
    plt.savefig(f"Run-code-dv-dc-k-n-label={code_class}-{dv}-{dc}-{k}-{n}-{label}{uid}/fer_plot.png")
    plt.show()

if __name__ == "__main__":
    with Profile() as prof:
        n_motifs, n_picks = 8, 4
        dv, dc, ffdim = 3, 9, 67
        k, n = 100 ,150
        L, M = 6, 6
        read_length = 6
        read_lengths = np.arange(1,12)
        run_fer(n_motifs, n_picks, dv, dc, k, n, L, M, ffdim, code_class="sc_", read_lengths=read_lengths, saved_code=False)
    (
        Stats(prof)
        .strip_dirs()
        .sort_stats("cumtime")
        .print_stats(10)
    )

    