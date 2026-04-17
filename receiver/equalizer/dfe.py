import numpy as np

class DFEEqualizer:
    def __init__(self, modulation_params, channel_model="rayleigh"):
        self.modulation_params = modulation_params
        self.channel_model = channel_model


        # Параметры DFE 
        self.sps = modulation_params.get("sps", 4)
        self.num_ff_taps = 20  
        self.num_fb_taps = 5          
        self.mu = 0.008              
        self.w_ff = None
        self.w_fb = None
        self.ff_buf = None
        self.fb_buf = None

    def _decide(self, y):

        return np.sign(np.real(y))

    def process_eq(self, rx_signal, tx_signal):
        
        h_est = self.estimator.h_rayleigh(tx_signal, rx_signal)

        h = h_est.astype(complex)

        h_matched = np.conj(h[::-1])
        if len(h_matched) > self.num_ff_taps:
            h_matched = h_matched[:self.num_ff_taps]
        else:
            h_matched = np.pad(h_matched, (0, self.num_ff_taps - len(h_matched)))
        self.w_ff = h_matched / (np.max(np.abs(h_matched)) + 1e-8)


        tail_start = self.sps
        tail = -h[tail_start : tail_start + self.num_fb_taps]
        if len(tail) > self.num_fb_taps:
            tail = tail[:self.num_fb_taps]
        else:
            tail = np.pad(tail, (0, self.num_fb_taps - len(tail)))
        self.w_fb = tail


        self.ff_buf = np.zeros(self.num_ff_taps, dtype=complex)
        self.fb_buf = np.zeros(self.num_fb_taps, dtype=complex)

        N = len(rx_signal)
        num_symbols = N // self.sps
        y_eq = np.zeros(num_symbols, dtype=complex)
        decisions = np.zeros(num_symbols, dtype=complex)
        errors = np.zeros(num_symbols, dtype=complex)

        symbol_idx = 0
        for n in range(N):
    
            self.ff_buf = np.roll(self.ff_buf, 1)
            self.ff_buf[0] = rx_signal[n]


            if n % self.sps == 0 and symbol_idx < num_symbols:
                ff_contrib = np.dot(self.w_ff, self.ff_buf)
                fb_contrib = np.dot(self.w_fb, self.fb_buf)
                y = ff_contrib + fb_contrib
                y_eq[symbol_idx] = y


                desired = self._decide(y)

                decisions[symbol_idx] = desired
                error = desired - y
                errors[symbol_idx] = error

                # LMS-адаптация
                self.w_ff += self.mu * error * np.conj(self.ff_buf)  
                self.w_fb += self.mu * error * np.conj(self.fb_buf)

                # Сдвиг решения в feedback
                self.fb_buf = np.roll(self.fb_buf, 1)
                self.fb_buf[0] = desired

                symbol_idx += 1

        return y_eq, decisions  