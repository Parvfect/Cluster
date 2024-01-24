

import sc_ldpc_protograph
import numpy as np

def get_vns(dv, dc, M, L):
    
    cns_per_pos = int(dv / dc * M)

    Harr = []
    for vn_position in range(L):   
        seed = vn_position * cns_per_pos
        vns = seed + sc_ldpc_protograph.gen_slots_from_position(dv, dc, M)
        Harr.append(vns)
        
    return np.array(Harr)

def get_Harr_sc_ldpc(L, M, dv=3, dc=9):
    """ Interface to get all the cns the vns are connected to and the number of connections per check node """

    Harr = get_vns(dv, dc, M, L)
    n = L*M
    cns_len = int((L + dv - 1) * dv / dc * M)
    k = int(n - cns_len)
    dcs = np.zeros(cns_len, dtype=int)
    dvs = [dv for i in range(L * M)] 

    
    Harr = Harr.flatten()
    for i in Harr:
        dcs[i] += 1

    # Will also need to get dc's and dv's to initialize the Tanner graph - which can be obtained from the array graph itself.

    return Harr, dvs, dcs, k, n

def get_dv_dc(dv, dc, k, n, Harr):
    
    cns_len = n-k
    dcs = np.zeros(cns_len, dtype=int)
    dvs = [dv for i in range(n)] 

    Harr = Harr.flatten()
    for i in Harr:
        dcs[i] += 1

    return dvs, dcs
