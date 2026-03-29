import numpy as np

class ZFEqualizer:

    def __init__(self, modulation_params, channel_model):
        self.sps = modulation_params.get("sps", 4)
        self.channel_model = channel_model

    def equalize(self, rx, h):

        H = np.fft.fft(h, len(rx))
        R = np.fft.fft(rx)
        
        S = R / (H + 1e-8)

        s = np.fft.ifft(S)

        return s

    def process_eq(self, match_signal, h):
        
        samples_per_burst = 156 * self.sps
        num_bursts = len(match_signal) // samples_per_burst

        eq_signal = []
        for b in range(num_bursts):
            start_idx = b * samples_per_burst
            end_idx = (b + 1) * samples_per_burst

            rx_burst = match_signal[start_idx  : end_idx]

            eq_burst= self.equalize(rx_burst, h[b])
            eq_signal.append(eq_burst)

        return np.concatenate(eq_signal)