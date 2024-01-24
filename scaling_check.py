
import row_echleon as r
import time
import matplotlib.pyplot as plt

dv, dc = 3,9
rate = 0.66

n = [21, 51 ,102, 201, 402, 801, 1200] #, 5001, 10002, 40002, 600000]
times = []

for i in n:
    startime = time.time()
    k = int(i*rate)
    Harr = r.get_H_arr(dv, dc, k, i)
    H = r.get_H_Matrix(dv, dc, k, i, Harr)
    #G = r.parity_to_generator(H, ffdim=ffdim)
    G = r.alternative_parity_to_generator(H, ffdim=67)
    times.append(time.time() - startime)


plt.plot(n, times)
plt.title("Time to create Generator Matrix for increasing codewords")
plt.xlabel("Length of codeword")
plt.ylabel("Time (s)")
plt.show()