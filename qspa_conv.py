from collections import defaultdict
from more_itertools import distinct_permutations

import galois
import numpy as np
from scipy.stats import multinomial
from math import factorial
import utils
import math


def multinomial_def(n, x, p):
    return (factorial(n)/(np.prod([factorial(i) for i in x])))*(np.prod([p[j]**x[j] for j in range(len(x))]))

def hw_likelihoods(k_motifs, codeword_noise, eps, threshold=1e10):
    """Get initial likelihoods for HelixWorks interference channel.

    Each symbol in the alphabet consists of a set of k_motifs out of n_motifs
    possible options. We represent a symbol as an array (x_1, ..., x_n_motifs)
    where xᵢ ∈ {0,1} and ∑ xᵢ = k_motifs. We consider a lexicographical order
    for all symbols X (written from left to right).

    The received codeword is represented as an array (y_1, ..., y_n_motifs)
    where ∑ yᵢ = R and R is the number of reads. The meaning of yᵢ is how many
    times (out of R in total) motif i was read. The likelihood P(Y | X) is
    computed using the formula for the multinomial PMF, without the factorials.

    Very small likelihoods are converted to 0, after which all likelihoods are
    scaled by the smallest one (in log domain). This is done to avoid extremely
    low values for the likelihoods which could result in numerical 0s.

    The strategy for deciding which likelihoods get converted to 0:
    (1) sort likelihoods
    (2) find largest consecutive likelihoods whose ratio > threshold
    (3) all smaller likelihoods are converted to 0

    Parameters
    ----------
    k_motifs: int
        Number of motifs that are chosen to create a "symbol".
    codeword_noise: array (n_motifs, )
        Stores the number of reads of each one of the n total motifs.
    eps: float
        Interference probability.
    threshold: float
        Minimum probability ratio spread to decide which likelihoods become 0.

    Returns
    -------
    likelihoods: array (n_motifs choose k_motifs, )
        Non-normalized likelihood for all symbols (in lexicographical order).

    >>> eps = 0.05
    >>> k_motifs = 3
    >>> codeword_noise = [1, 6, 9, 3, 4]
    >>> hw_likelihoods_v2(k_motifs, codeword_noise, eps, 10000)
    """
    n_motifs = len(codeword_noise)
    R = sum(codeword_noise)

    prob_base = eps / n_motifs
    prob_high = (1 - eps) / k_motifs

    alphabet = distinct_permutations(
        [0] * (n_motifs - k_motifs) + [1] * k_motifs,
        r=n_motifs
    )
    log_likelihoods = []
    for symbol in alphabet:
        reads_high = sum(np.array(symbol) * codeword_noise)
        reads_base = R - reads_high

        # Subtract maximum common likelihood
        m = min(codeword_noise)
        reads_high = reads_high - m * k_motifs
        reads_base = reads_base - m * (n_motifs - k_motifs)
        
        # Compute log-likelihood
        log_likelihoods.append(
            reads_base * math.log2(prob_base)
            + reads_high * math.log2(prob_base + prob_high)
        )

    idx_original = np.argsort(np.argsort(log_likelihoods))
    log_likelihoods = np.sort(log_likelihoods)

    i = 0
    max_idx_zero = 0
    threshold = math.log2(threshold)
    while i < len(log_likelihoods) - 1:
        if log_likelihoods[i + 1] - log_likelihoods[i] > threshold:
            max_idx_zero = i + 1
        i += 1

    log_likelihoods = log_likelihoods - log_likelihoods[max_idx_zero]
    likelihoods = 2 ** log_likelihoods
    likelihoods[0:max_idx_zero] = 0

    likelihoods = likelihoods[idx_original]
    likelihoods = np.flip(likelihoods)

    return list(likelihoods / sum(likelihoods))


def hw_likelihoods_old(k_motifs, codeword_noise, eps):
    """Get initial likelihoods for HelixWorks interference channel.

    Each symbol in the alphabet consists of a set of k_motifs out of n_motifs
    possible options. We represent a symbol as an array (x_1, ..., x_n_motifs)
    where xᵢ ∈ {0,1} and ∑ xᵢ = k_motifs. We consider a lexicographical order
    for all symbols X (written from left to right).

    The received codeword is represented as an array (y_1, ..., y_n_motifs)
    where ∑ yᵢ = R and R is the number of reads. The meaning of yᵢ is how many
    times (out of R in total) motif i was read. The likelihood P(Y | X) is
    computed using a multinomial PMF.

    Parameters
    ----------
    k_motifs: int
        Number of motifs that are chosen to create a "symbol".
    codeword_noise: array (n_motifs, )
        S[i, j, :] stores messages from CN i to VN j.
    eps: float
        Interference probability.

    Returns
    -------
    likelihoods: array (n_motifs choose k_motifs, )
        Non-normalized likelihood for each symbols (in lexicographical order).

    >>> eps = 0.05
    >>> k_motifs = 2
    >>> codeword_noise = [10, 1, 7, 2]
    >>> hw_likelihoods(k_motifs, codeword_noise, eps)
    """
    n_motifs = len(codeword_noise)
    R = sum(codeword_noise)

    prob_base = np.ones(n_motifs) * eps / n_motifs
    prob_high = (1 - eps) / k_motifs

    alphabet = distinct_permutations(
        [0] * (n_motifs - k_motifs) + [1] * k_motifs,
        r=n_motifs
    )
    likelihoods = []
    for symbol in alphabet:
        likelihoods.append(multinomial_def(n=R, x =codeword_noise, p=prob_base + np.array(symbol) * prob_high))

    likelihoods.reverse()
    return likelihoods


class QSPADecoder:
    """Class implementing QSPA Decoder described in [1].
    
    [1] Ryan, William, and Shu Lin. Channel Codes: Classical an` Modern (2009).
    """

    def __init__(self, n, m, GF, GFH):
        self.n = n      # length of codeword
        self.m = m      # number of parity-check constraints
        self.GF = GF    # Galois field
        self.GFH = GFH  # parity-check matrix (with elements in GF)

        self.nonzero_cols, self.nonzero_rows = self.index_nonzero()

        # Store additive inverse for each element in GF
        self.idx_shuffle = np.array([
            (GF.order - a) % GF.order for a in range(GF.order)
        ])

    def index_nonzero(self):
        """
        Data structures to store non-zero entries of GFH.

        Returns
        -------
        nonzero_cols: dict (int -> list)
            For row i, return columns j such that GFH[i, j] ≠ 0.
        nonzero_rows: dict (int -> list)
            For column j, return rows i such that GFH[i, j] ≠ 0.
        """
        nonzero_cols = defaultdict(list)
        nonzero_rows = defaultdict(list)
        for i in range(self.m):
            idxs = np.nonzero(self.GFH[i, :])[0]
            for j in idxs:
                nonzero_cols[i].append(j)
                nonzero_rows[j].append(i)
        for k, v in nonzero_cols.items():
            nonzero_cols[k] = np.array(v)
        for k, v in nonzero_rows.items():
            nonzero_rows[k] = np.array(v)
        return nonzero_cols, nonzero_rows

    def decode(self, P, max_iter=50):
        """QSPA Decoder main loop.
        
        Parameters
        ----------
        P: array (n, GF.order)
            Initial likelihoods for each symbol in received codeword.
        max_iter: int
            Maximum number of iterations that decoder should run for.

        Returns
        -------
        z: GF array (n, )
            Codeword such that GFH * z = 0, if decoding is successful.
        """

        Q = np.zeros(shape=(self.m, self.n, self.GF.order))
        S = np.zeros(shape=(self.m, self.n, self.GF.order))

        # Initializes Variable Nodes with the Likelihoods
        Q = self.initialize_Q_msgs(P, Q)
        
        prev_z = self.decode_hard(P,S)

        for it in range(max_iter):
            #print(f'Decoding: iteration {it + 1}')
            S = self.update_S_msgs(Q, S)
            z = self.decode_hard(P, S)
            
            parity = not np.matmul(self.GFH, z).any()
            if parity:
                print(f'Decoding successful! Iteration {it + 1}')
                return z
            else:
                Q = self.update_Q_msgs(P, Q, S)

            if np.array_equal(z, prev_z):
                break
            
            prev_z = z
        #print('Decoding unsuccessful! Max. iterations done')

    def initialize_Q_msgs(self, P, Q):
        """Initialize messages from variable nodes (VN) to check nodes (CN).

        Parameters
        ----------
        P: array (n, GF.order)
            Initial likelihoods for each symbol in received codeword.
        Q: array (m, n, GF.order)
            Q[i, j, :] stores messages from VN j to CN i, starts as zero.

        Returns
        -------
        Q: array (m, n, GF.order)
            Initial messages (i.e., likelihoods) from VNs to CNs.
        """
        for j in range(self.n):
            idxs = self.nonzero_rows[j]
            for i in idxs:
                Q[i, j, :] = 1 * P[j, :]
        return Q

    def update_S_msgs(self, Q, S):
        """Update S messages following algorithm in [1]."""
       
        #Q_ = self._shift_Q_msgs(Q)
        Q_ = Q

        S_ = np.zeros(shape=(self.m, self.n, self.GF.order))
        for i in range(self.m):
            idxs = self.nonzero_cols[i]

            for j in idxs:

                # Remove val equivalent
                conv_idxs = idxs[idxs != j]

                # For some reason does first convolution
                aux = self._conv_circ(
                    Q_[i, conv_idxs[0], :],
                    Q_[i, conv_idxs[1], :]
                )

                # Rest of the convolutions
                for t in conv_idxs[2:]:
                    aux = self._conv_circ(aux, Q_[i, t, :])
                
                # Becomes the additive inverse of that I is confused
                S_[i, j, :] = aux[self.idx_shuffle]

        #S = self._shift_S_msgs(S, S_)
        return S_

    def update_Q_msgs(self, P, Q, S):
        """Update Q messages following algorithm in [1]."""

        # Got to understand the parity equivalent
        for a in range(self.GF.order):
            for j in range(self.n):
                idxs = self.nonzero_rows[j]
                for i in idxs:
                    # Initial Likelihoods
                    Q[i, j, a] = 1 * P[j, a]

                    # Don't understand this step - has to do with CN update
                    for t in idxs[idxs != i]:
                        Q[i, j, a] *= S[t, j, a]

                    # Normalization
                    Q[i, j, :] /= sum(Q[i, j, :])
        return Q

    def decode_hard(self, P, S):
        """Get pseudo maximum likelihood codeword.

        Parameters
        ----------
        P: array (n, GF.order)
            Initial likelihoods for each symbol in received codeword.
        S: array (m, n, GF.order)
            S[i, j, :] stores messages from CN i to VN j.

        Returns
        -------
        z: GF array (n, )
            Most likely codeword in light of likelihoods and check messages.
        """
        z = np.zeros(self.n)
        for j in range(self.n):
            idxs = self.nonzero_rows[j]
            probs = 1 * P[j, :]
            for a in range(self.GF.order):
                for i in idxs:
                    probs[a] *= S[i, j, a]
            z[j] = np.argmax(probs)
        z = self.GF(z.astype(int))
        return z

    def _shift_Q_msgs(self, Q):
        """Re-order indices in Q following FFT QSPA algorithm in [1].

        Parameters
        ----------
        Q: array (m, n, GF.order)
            Q[i, j, :] stores messages from VN j to CN i.

        Returns
        -------
        Q_: array (m, n, GF.order)
            Same as Q, with Q[i, j, :] re-ordered for all (i, j).
        """
        Q_ = np.zeros(shape=(self.m, self.n, self.GF.order))
        for i in range(self.m):
            idxs = self.nonzero_cols[i]
            for j in idxs:
                for a in range(self.GF.order):
                    Q_[i, j, self.GFH[i, j] * self.GF(a)] = 1 * Q[i, j, a]
        return Q_

    def _shift_S_msgs(self, S, S_):
        """Re-order indices in S following FFT QSPA algorithm in [1].

        Parameters
        ----------
        S: array (m, n, GF.order)
            S[i, j, :] stores messages from CN i to VN j.
        S_: array (m, n, GF.order)
            Same as S, with S[i, j, :] re-ordered for all (i, j).

        Returns
        -------
        S: array (m, n, GF.order)
            S with indices back to normal.
        """
        for i in range(self.m):
            idxs = self.nonzero_cols[i]
            for j in idxs:
                for a in range(self.GF.order):
                    S[i, j, a] = 1 * S_[i, j, self.GFH[i, j] * self.GF(a)]
        return S

    @staticmethod
    def _conv_circ(u, v):
        """Perform circular convolution between u and v over GF using FFT."""
        return np.real(np.fft.ifft(np.fft.fft(u) * np.fft.fft(v)))


def test():
    """
    Test QSPA decoder for a simple channel that adds +1 to each symbol in
    codeword with probability 1 - eps = 0.05 in example.
    """
    n_code = 16
    GF = galois.GF(3)

    # Get binary parity-check matrix using Gallager's algorithm.
    H = utils.parity_check_matrix(n_code, d_v=3, d_c=4)
    m_checks = H.shape[0] # n-k = m - number of check nodes
    density = sum(sum(H)) / (H.shape[0] * H.shape[1])
    print(f'Density of parity-check matrix: {density}')

    # Turn binary matrix into matrix over GF field
    GFH = GF(H) * GF(np.random.choice(GF.elements[1:], size=H.shape))
    GFK = GFH.null_space()
    GFK_dim = GFK.shape[0]
    print(f'Code rate: {GFK_dim / n_code}')

    # Create random codeword and transmit with random noise over channel
    print(f'Testing for a simple +1 channel')
    eps = 0.95
    codeword = np.matmul(GFK.T, GF.Random(GFK_dim))

    def transmit(w):
        noise = GF((np.random.uniform(0, 1, n_code) > eps).astype(int))
        return w + noise

    codeword_noise = transmit(codeword)
    while np.array_equal(codeword, codeword_noise):
        codeword_noise = transmit(codeword)

    '''
    Decode codeword_noise.
    Interesting example: 8 iterations until convergence for
    >> codeword
    GF([2, 1, 1, 1, 2, 0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 1], order=3)
    >> codeword_noise
    GF([0, 2, 1, 1, 2, 0, 0, 2, 2, 2, 2, 1, 1, 0, 2, 1], order=3)
    '''
    P = []
    for a in codeword_noise:
        base = np.zeros(GF.order)
        base[a] = eps
        base[a - GF(1)] = 1 - eps
        P.append(base)
    P = np.array(P)

    decoder = QSPADecoder(n_code, m_checks, GF, GFH)
    z = decoder.decode(P, max_iter=10)
    assert np.array_equal(codeword, z)


if __name__ == '__main__':
    test()
