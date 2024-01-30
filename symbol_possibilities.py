
from itertools import combinations
import numpy as np

def choose_symbols(n_motifs, picks):
    """ Returns Symbol Dictionary given the motifs and the number of picks """
    return [list(i) for i in (combinations(np.arange(1, n_motifs+1), picks))]

def get_symbol_possibilites_precomputed():
    motifs = np.arange(1,9)

    symbols = choose_symbols(8, 4)
    
    symbols.pop(-1)
    symbols.pop(-2)
    symbols.pop(-3)

    # Precomputing symbol possibililities
    symbol_possibilities = dict()
    for i in motifs:
        motif_arr = [j for j in range(1,9) if not j == i]
        remaining_motif_combinations = [list(i) for i in (combinations(motif_arr, 3))]
        motif_combinations = [[i]+ j  for j in remaining_motif_combinations]
        for motif_combination in motif_combinations:
            motif_combination.sort()
        symbol_possibilities_ = np.array([symbols.index(j) for j in motif_combinations if j in symbols])
        symbol_possibilities[i] = set(symbol_possibilities_)

    return symbol_possibilities