import numpy as np

class ViterbiDecoder:

    def __init__(self, constraint_length, polynomials, mode="vit_hard"):

        self.K = constraint_length
        self.polynomials = polynomials
        self.mode = mode

        self.n_outputs = len(polynomials)
        self.n_states = 2 ** (constraint_length - 1)

        self.next_state = np.zeros((self.n_states, 2), dtype=int)
        self.output = np.zeros((self.n_states, 2, self.n_outputs), dtype=int)

        self._build_trellis()


    def _poly_to_bits(self, poly):
        return [(poly >> (self.K-1-i)) & 1 for i in range(self.K)]


    def _build_trellis(self):

        poly_bits = [self._poly_to_bits(p) for p in self.polynomials]

        for state in range(self.n_states):

            reg = [(state >> i) & 1 for i in range(self.K - 1)]

            for bit in (0, 1):

                shift_reg = [bit] + reg

                out_bits = []

                for poly in poly_bits:

                    val = 0
                    for i in range(self.K):
                        if poly[i]:
                            val ^= shift_reg[i]

                    out_bits.append(val)

                ns = ((state << 1) | bit) & (self.n_states - 1)

                self.next_state[state, bit] = ns
                self.output[state, bit] = out_bits


    def decode(self, coded_bits):

        coded_bits = np.array(coded_bits)
        coded_bits = np.nan_to_num(coded_bits, nan=0.0, posinf=20.0, neginf=-20.0)
        coded_bits = np.clip(coded_bits, -20, 20)

        n_steps = len(coded_bits) // self.n_outputs
        coded_bits = coded_bits.reshape((n_steps, self.n_outputs))

        path_metric = np.full(self.n_states, np.inf)
        path_metric[0] = 0

        prev_state = np.zeros((n_steps, self.n_states), dtype=int)
        prev_bit = np.zeros((n_steps, self.n_states), dtype=int)

        for t in range(n_steps):

            new_metric = np.full(self.n_states, np.inf)
            r = coded_bits[t]

            for state in range(self.n_states):

                if path_metric[state] == np.inf:
                    continue

                for bit in (0, 1):

                    ns = self.next_state[state, bit]
                    expected = self.output[state, bit]
                    
                    if self.mode == "vit_hard" or self.mode == "diff_phase":
                        dist = np.sum(r != expected)
                        
                    elif self.mode == "vit_soft":
                        dist = -np.sum((1 - 2*np.array(expected)) * r)
                        
                    metric = path_metric[state] + dist

                    if metric < new_metric[ns]:

                        new_metric[ns] = metric
                        prev_state[t, ns] = state
                        prev_bit[t, ns] = bit

            path_metric = new_metric


        state = np.argmin(path_metric)  #  encoder terminates to zero state

        decoded = []

        for t in reversed(range(n_steps)):

            bit = prev_bit[t, state]
            decoded.append(bit)

            state = prev_state[t, state]

        decoded.reverse()

        return decoded