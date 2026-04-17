import numpy as np
from core.block import Block


class MatchedFilter(Block):

    def __init__(self, modulation_params, is_working=False):
        super().__init__(is_working)

        self.sps = modulation_params.get("sps", 4)

    def _process(self, rx_signal, h):
        
        samples_per_burst = 156 * self.sps
        num_bursts = len(rx_signal) // samples_per_burst

        match_signal = []

        for b in range(num_bursts):
            h_est = h[b]
            start_idx = b * samples_per_burst
            end_idx = (b + 1) * samples_per_burst
    
            rx_burst = rx_signal[start_idx  : end_idx]

            burst_match = np.convolve(rx_burst, np.conj(h_est[::-1]))
            burst_trunc = burst_match[int(h_est.size / 2) - 1: - int(h_est.size / 2)]
            match_signal.append(burst_trunc)

        return np.concatenate(match_signal)
