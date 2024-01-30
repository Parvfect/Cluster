
from itertools import combinations
import numpy as np

motifs = np.arange(1,9)

symbols = [list(i) for i in (combinations(motifs, 4))]
print(symbols)

# Precomputing symbol possibililities
single_motif = motifs.copy()
symbol_possibilities = dict()
for i in motifs:
    motif_arr = [j for j in range(1,9) if not j == i]
    remaining_motif_combinations = [list(i) for i in (combinations(motif_arr, 3))]
    motif_combinations = [[i]+ j  for j in remaining_motif_combinations]
    for i in motif_combinations:
        i.sort()
    symbol_possibilities_ = [symbols.index(j) for j in motif_combinations]
    symbol_possibilities[i] = symbol_possibilities_

print(symbol_possibilities)