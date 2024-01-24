
from protograph_interface import get_Harr_sc_ldpc
import row_echleon as r
import uuid
import os
import numpy as np

dv, dc, k, n, ffdim = 3, 9, 852, 1278, 67
Harr, dvs, dcs, k, n, L, M = get_Harr_sc_ldpc()

H = r.get_H_matrix_sclpdc(dcs, dvs, k, n, Harr)
dc = max(dcs)
dv = dvs[0]
G = r.parity_to_generator(H, ffdim=ffdim)

if np.any(np.dot(G, H.T) % ffdim != 0):
    print("Matrices are not valid, aborting simulation")
    exit()

unique_filename = str(uuid.uuid4())

filename = "codes/sc_dv_dc_k_n_L_M_ffdim={}_{}_{}_{}_{}_{}_{}/{}".format(dv, dc, k, n, L, M, ffdim, unique_filename)

# Create the directory if it does not exist
if not os.path.exists(filename):
    os.makedirs(filename)

# Save the Harr, H and G matrices
np.save(filename + "/Harr", Harr)
np.save(filename + "/H", H)
np.save(filename + "/G", G)

print("Saved the matrices to {}".format(filename))