import numpy as np

class ViterbiDecoderACS:
    
    def __init__(self, constraint_length: int, polynomials: list):
        self.K = constraint_length
        self.polynomials = polynomials
        self.n_outputs = len(polynomials)
        self.n_states = 2 ** (constraint_length - 1)

        self.next_state = np.zeros((self.n_states, 2), dtype=int)
        self.output = np.zeros((self.n_states, 2, self.n_outputs), dtype=int)

        self._build_trellis()

    def _poly_to_bits(self, poly):
        return [(poly >> (self.K - 1 - i)) & 1 for i in range(self.K)]

    def _build_trellis(self):
        poly_bits = [self._poly_to_bits(p) for p in self.polynomials]

        for state in range(self.n_states):
            reg = [(state >> i) & 1 for i in range(self.K - 1)]
            for bit in (0, 1):
                shift_reg = [bit] + reg
                out_bits = [sum(p[i] * shift_reg[i] for i in range(self.K)) % 2 
                           for p in poly_bits]

                ns = ((state << 1) | bit) & (self.n_states - 1)
                self.next_state[state, bit] = ns
                self.output[state, bit] = out_bits

    def _compute_branch_metric(self, r: np.ndarray, expected: np.ndarray) -> float:
        expected = np.asarray(expected)
        return -np.sum((1 - 2 * expected) * r)

    def decode(self, llr_list: list):

        if len(llr_list) < 2:
            raise ValueError("Должно быть минимум 2 сектора")

        llrs = [np.asarray(llr, dtype=np.float64) for llr in llr_list]
        

        n_steps = len(llrs[0]) // self.n_outputs
        n_sectors = len(llrs)

        r = np.stack([llr.reshape((n_steps, self.n_outputs)) for llr in llrs])
        
        path_metric = np.full(self.n_states, np.inf)
        path_metric[0] = 0.0

        prev_state = np.zeros((n_steps, self.n_states), dtype=int)
        prev_bit = np.zeros((n_steps, self.n_states), dtype=int)
        prev_sector = np.zeros((n_steps, self.n_states), dtype=int)  

        for t in range(n_steps):
            new_metric = np.full(self.n_states, np.inf)
            r_t = r[:, t, :]                     # разделение на n_sectors, n_outputs

            for state in range(self.n_states):
                if path_metric[state] == np.inf:
                    continue

                for bit in (0, 1):
                    ns = self.next_state[state, bit]
                    expected = self.output[state, bit]

                    
                    branch_metrics = np.array([
                        self._compute_branch_metric(r_t[s], expected) 
                        for s in range(n_sectors)
                    ])   

    
                    candidate_metrics = path_metric[state] + branch_metrics

                    best_idx = np.argmin(candidate_metrics)
                    best_metric = candidate_metrics[best_idx]

                    if best_metric < new_metric[ns]:
                        new_metric[ns] = best_metric
                        prev_state[t, ns] = state
                        prev_bit[t, ns] = bit
                        prev_sector[t, ns] = best_idx   

            path_metric = new_metric

        state = np.argmin(path_metric)
        decoded = []

        for t in reversed(range(n_steps)):
            bit = prev_bit[t, state]
            decoded.append(bit)
            state = prev_state[t, state]

        decoded.reverse()
        
        return decoded