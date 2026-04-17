import numpy as np
from core.block import Block


class Modulation(Block):

    def __init__(self, scheme, params, is_working=False):
        super().__init__(is_working)

        if scheme in ["TCHFS", "CS1", "MCS1"]:

            self.modulator = GMSKModulation(params)

        elif scheme == "MCS5":

            self.modulator = PSKModulation(params)

        else:

            raise ValueError("Unknown scheme")

    def _process(self, bits):

        return self.modulator.process_mod(bits)


class GMSKModulation:

    def __init__(self, params):
        self.BT = params.get("BT", 0.3)
        self.T = params.get("T", 3.69e-6)
        self.sps = params.get("sps", 4)
        self.dt = self.T / self.sps
        self.h = params.get("h", 0.5)
        self.gaus_duration = params.get("gaus_duration", 3)
        self.rect_duration = params.get("rect_duration", 1)

        return

    def differential_encoding(self, bits):
        if bits.size % 148 != 0:
            raise ValueError("Количество модуляционных бит должно быть кратным 148")

        d_prev = np.ones(bits.size, dtype=int)
        d_prev[1:] = bits[:-1]

        d_curr = bits ^ d_prev
        alpha = 1 - 2 * d_curr

        return alpha

    def generate_q_gmsk(self):
        BT = self.BT
        T = self.T
        gaus_duration = self.gaus_duration
        rect_duration = self.rect_duration

        oversampling = 100
        sps_oversampling = self.sps * oversampling
        dt_oversampling = T / sps_oversampling

        delta = np.sqrt(np.log(2)) / (2 * np.pi * BT)

        t_h = np.arange(-gaus_duration * (T / 2), gaus_duration * (T / 2), dt_oversampling)
        t_rect = np.arange(-rect_duration * (T / 2), rect_duration * (T / 2), dt_oversampling)

        h_t = np.exp(-(t_h**2) / (2 * (delta**2) * (T**2))) / (
            np.sqrt(2 * np.pi) * delta * T
        )
        rect = np.ones(t_rect.size) / T

        g_t = np.convolve(h_t, rect) * dt_oversampling

        q_gmsk_oversampling = np.cumsum(g_t) * dt_oversampling
        q_gmsk = q_gmsk_oversampling[::oversampling]

        return q_gmsk

    def calc_phase(self, alpha, q_gmsk):
        h = self.h
        dt = self.dt
        sps = self.sps
        gaus_duration = self.gaus_duration
        rect_duration = self.rect_duration

        alpha_repeat = np.repeat(alpha, sps)

        num_bits = alpha.size
        phi = np.zeros(num_bits * sps + q_gmsk.size - sps)

        for i in range(num_bits):
            alpha_i = alpha[i]
            start_idx = i * sps

            phi[start_idx : start_idx + q_gmsk.size] += alpha_i * np.pi * h * q_gmsk

            phase_step = alpha_i * np.pi * h
            phi[start_idx + q_gmsk.size :] += phase_step

        # Сдвиг, чтобы изменению символа соответствовало изменение фазы
        shift = (gaus_duration + rect_duration) / 2 - 0.5
        phi_shift = phi[int(shift * sps) : int(shift * sps) + alpha_repeat.size]

        return phi_shift

    def process_mod(self, bits):
        
        # Делим на 148, а не 156, что бы без кодера тоже работало
        # Так как берем целую часть, то на результат не влияет 
        active_size = 148
        num_bursts = len(bits) // active_size
    
        q_gmsk = self.generate_q_gmsk()
        
        all_signals = []
        guard_period = np.zeros(8 * self.sps, dtype=complex)
        gp_len = 0
        for i in range(num_bursts):
            burst = bits[gp_len + i*active_size : gp_len + (i+1)*active_size]
            gp_len += 8

            alpha = self.differential_encoding(burst)
            phi = self.calc_phase(alpha, q_gmsk)
            signal_envelope = np.exp(1j * phi)
            
            all_signals.append(signal_envelope)
            all_signals.append(guard_period)

        return np.concatenate(all_signals)



class PSKModulation:
    def __init__(self, params):
        raise ValueError("Еще не реализован")
