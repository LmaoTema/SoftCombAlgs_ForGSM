import numpy as np
from .vit_detector_osmo import calc_increment_osmo, calc_metric_osmo, find_best_stop_state_osmo, traceback_osmo

class GMSKDetector:
    def __init__(self, params, block_params):
        self.BT = params.get("BT", 0.3)
        self.T = params.get("T", 3.69e-6)
        self.sps = params.get("sps", 4)
        self.dt = self.T / self.sps
        self.h = params.get("h", 0.5)
        self.gaus_duration = params.get("gaus_duration", 4)
        self.rect_duration = params.get("rect_duration", 1)
        self.type_demod = params.get("type_demod", "diff_phase") # diff_phase / vit_hard / vit_soft 

        self.mf_is_working = block_params["matched filter"]["is_working"]

    def calc_rhh(self, h):
        n = np.arange(h.size)
        # Разобраться с инкрементами. Надо ли вращать самостоятельно
        # h_complex = h * (1j**(n / self.sps))
        h_complex = h
        rhh_full = np.convolve(h_complex, np.conj(h_complex[::-1]))
        center_idx = h_complex.size - 1
        rhh = rhh_full[center_idx :: self.sps]

        return rhh

    def calc_increment(self, rhh):
        # Определяем влияние предыдущих бит для каждого состояния
        # C учетом деротации (+ - - +)
        increment = np.zeros(16)
        increment[0] = rhh[4].real + rhh[3].real + rhh[2].real + rhh[1].real
        increment[1] = rhh[4].real + rhh[3].real + rhh[2].real - rhh[1].real
        increment[2] = rhh[4].real + rhh[3].real - rhh[2].real + rhh[1].real
        increment[3] = rhh[4].real + rhh[3].real - rhh[2].real - rhh[1].real
        increment[4] = rhh[4].real - rhh[3].real + rhh[2].real + rhh[1].real
        increment[5] = rhh[4].real - rhh[3].real + rhh[2].real - rhh[1].real
        increment[6] = rhh[4].real - rhh[3].real - rhh[2].real + rhh[1].real
        increment[7] = rhh[4].real - rhh[3].real - rhh[2].real - rhh[1].real
        increment[8] = - increment[7]
        increment[9] = - increment[6]
        increment[10] = - increment[5]
        increment[11] = - increment[4]
        increment[12] = - increment[3]
        increment[13] = - increment[2]
        increment[14] = - increment[1]
        increment[15] = - increment[0]
        # increment[0] = rhh[4].real - rhh[3].imag - rhh[2].real + rhh[1].imag
        # increment[1] = rhh[4].real - rhh[3].imag - rhh[2].real - rhh[1].imag
        # increment[2] = rhh[4].real - rhh[3].imag + rhh[2].real + rhh[1].imag
        # increment[3] = rhh[4].real - rhh[3].imag + rhh[2].real - rhh[1].imag
        # increment[4] = rhh[4].real + rhh[3].imag - rhh[2].real + rhh[1].imag
        # increment[5] = rhh[4].real + rhh[3].imag - rhh[2].real - rhh[1].imag
        # increment[6] = rhh[4].real + rhh[3].imag + rhh[2].real + rhh[1].imag
        # increment[7] = rhh[4].real + rhh[3].imag + rhh[2].real - rhh[1].imag
        # increment[8] = - increment[7]
        # increment[9] = - increment[6]
        # increment[10] = - increment[5]
        # increment[11] = - increment[4]
        # increment[12] = - increment[3]
        # increment[13] = - increment[2]
        # increment[14] = - increment[1]
        # increment[15] = - increment[0]

        return increment

    def calc_metric(self, increment, sampled_signal, start_state=0):
        # Расчет метрик для всех возможных состояний
        old_path_metrics = np.ones(16) * -1e30
        old_path_metrics[start_state] = 0.0
        new_path_metrics = np.zeros(16)

        samples_num = sampled_signal.size

        trans_table = np.zeros((samples_num, 16))

        sample_nr = 0
        # Учет деротации (+ - - +)
        sign_rotate = 1

        while sample_nr < samples_num:
            
            if (sample_nr % 2) == 0:
                input_symbol =  sign_rotate * sampled_signal[sample_nr].imag
            else:
                sign_rotate = - sign_rotate
                input_symbol =  sign_rotate * sampled_signal[sample_nr].real

            # в итоге инвертированный знак. может и инкременты поменять
            for i in range(8):
                pm_candidate1 = old_path_metrics[i] + input_symbol - increment[i]
                pm_candidate2 = old_path_metrics[i + 8] + input_symbol - increment[i + 8]
                paths_difference = pm_candidate2 - pm_candidate1
                if paths_difference < 0:
                    new_path_metrics[2 * i] = pm_candidate1
                else:
                    new_path_metrics[2 * i] = pm_candidate2
                trans_table[sample_nr][2 * i] = paths_difference

                pm_candidate1 = old_path_metrics[i] - input_symbol + increment[i]
                pm_candidate2 = old_path_metrics[i + 8] - input_symbol + increment[i + 8]
                paths_difference = pm_candidate2 - pm_candidate1
                if paths_difference < 0:
                    new_path_metrics[2 * i + 1] = pm_candidate1
                else:
                    new_path_metrics[2 * i + 1] = pm_candidate2
                trans_table[sample_nr][2 * i + 1] = paths_difference

            tmp = new_path_metrics
            new_path_metrics = old_path_metrics
            old_path_metrics = tmp

            sample_nr += 1

        return trans_table, old_path_metrics
    
    def find_best_stop_state(self, old_path_metrics, stop_states=[0, 8]):
        best_stop_state = stop_states[0]
        max_stop_state_metric = old_path_metrics[best_stop_state]
        for s in stop_states:
            if old_path_metrics[s] > max_stop_state_metric:
                max_stop_state_metric = old_path_metrics[s]
                best_stop_state = s

        return best_stop_state
    
    def traceback(self, trans_table, best_stop_state):
        state_transfer = [
            [0, 8],
            [0, 8],
            [1, 9],
            [1, 9],
            [2, 10],
            [2, 10],
            [3, 11],
            [3, 11],
            [4, 12],
            [4, 12],
            [5, 13],
            [5, 13],
            [6, 14],
            [6, 14],
            [7, 15],
            [7, 15]
        ]

        bits = np.zeros(148)
        sample_nr = 148
        curr_state = best_stop_state
        while sample_nr > 0:
            sample_nr -= 1
            paths_difference = trans_table[sample_nr][curr_state]
            
            if (self.type_demod == "vit_hard"):
                if paths_difference > 0:
                    prev_state = state_transfer[curr_state][1]
                else:
                    prev_state = state_transfer[curr_state][0]

                bits[sample_nr] = (curr_state % 2) ^ 1

            curr_state = prev_state

        return bits

    def diff_phase(self, burst_samples):
        y_k = burst_samples[self.sps - 1 :: self.sps]

        y_k_prev = np.zeros(y_k.size, dtype=complex)
        y_k_prev[1:] = y_k[:-1]
        y_k_prev[0] = 1 + 0j

        delta_phi = np.angle(y_k * np.conj(y_k_prev))

        alpha = np.ones(delta_phi.size)
        alpha[delta_phi <= 0] = -1

        d_curr = ((1 - alpha) / 2).astype(int)

        burst_bits = np.zeros(d_curr.size, dtype=int)
        d_prev = 1
        for i in range(d_curr.size):
            burst_bits[i] = d_curr[i] ^ d_prev
            d_prev = burst_bits[i]

        return burst_bits

    def process_detect(self, complex_signal, h):
        
        sps = self.sps
        samples_per_burst = 156 * sps
        num_bursts = len(complex_signal) // samples_per_burst
    
        all_bits = []

        for b in range(num_bursts):
            if self.type_demod in ["vit_soft", "vit_hard"]:
                if self.mf_is_working == False:
                    increment = np.zeros(16)
                else:
                    rhh = self.calc_rhh(h[b])
                    increment = self.calc_increment(rhh)
                    # increment = np.zeros(16)
        
            start_idx = b * samples_per_burst
            burst_samples = complex_signal[start_idx : start_idx + 148 * sps]

            if self.type_demod == "diff_phase":
                burst_bits = self.diff_phase(burst_samples)
                all_bits.append(burst_bits)

            elif self.type_demod in ["vit_soft", "vit_hard"]:
                sampled_signal = burst_samples[self.sps - 1 :: self.sps]

                trans_table, old_path_metrics = self.calc_metric(increment, sampled_signal, start_state=0)

                best_stop_state = self.find_best_stop_state(old_path_metrics)

                burst_bits = self.traceback(trans_table, best_stop_state)

                all_bits.append(burst_bits)
                
        bits = np.concatenate(all_bits)

        return bits