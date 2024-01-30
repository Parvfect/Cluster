
from itertools import combinations
import numpy as np


def get_symbol_possibilites_precomputed():
    motifs = np.arange(1,9)

    symbols = [list(i) for i in (combinations(motifs, 4))]

    # Precomputing symbol possibililities
    symbol_possibilities = dict()
    for i in motifs:
        motif_arr = [j for j in range(1,9) if not j == i]
        remaining_motif_combinations = [list(i) for i in (combinations(motif_arr, 3))]
        motif_combinations = [[i]+ j  for j in remaining_motif_combinations]
        for motif_combination in motif_combinations:
            motif_combination.sort()
        symbol_possibilities_ = np.array([symbols.index(j) for j in motif_combinations])
        symbol_possibilities_ = symbol_possibilities_[symbol_possibilities_ < 67]    
        print(symbol_possibilities_)
        symbol_possibilities[i] = set(symbol_possibilities_)

    return symbol_possibilities