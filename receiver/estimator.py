import numpy as np
import matplotlib.pyplot as plt
from transmitter.modulator import GMSKModulation
class ChannelEstimate():

    def __init__(self, modulation_params, simulation_params):
        self.BT = modulation_params.get("BT", 0.3)
        self.T = modulation_params.get("T", 3.69e-6)
        self.sps = modulation_params.get("sps", 4)
        self.gaus_duration = modulation_params.get("gaus_duration", 4)
        self.rect_duration = modulation_params.get("rect_duration", 1)
        # L - длина ИХ, а не глубина МСИ
        self.L = (self.gaus_duration + self.rect_duration)

        self.channel_model = simulation_params.get("channel_model", "awgn")
        self.force_training_estimation_awgn = simulation_params.get(
            "force_training_estimation_awgn", True
        )
        self.h = modulation_params.get("h", 0.5)
        self.est_channel_len_sps = modulation_params.get(
            "est_channel_len_sps", 5 * self.sps
        )
        self.estimator_reg = modulation_params.get("estimator_reg", 1e-2)
        self.debug_first_burst = modulation_params.get("debug_first_burst", True)
    
    # h(t) композитного фильтра (передатчик + канал)
    def h_awgn(self):
        BT = self.BT
        T = self.T
        gaus_duration = self.gaus_duration
        rect_duration = self.rect_duration
        L = self.L

        oversampling = 100
        sps_oversampling = self.sps * oversampling
        dt_oversampling = T/sps_oversampling

        delta = np.sqrt(np.log(2)) / (2 * np.pi * BT)

        t_h = np.arange(-gaus_duration / 2 * T, gaus_duration / 2 * T, dt_oversampling)
        t_rect = np.arange(-rect_duration / 2 * T, rect_duration / 2 * T, dt_oversampling)

        # Формируем гауссовский и прямоугольный импульсы
        h_t = np.exp(-(t_h**2) / (2 * (delta**2) * (T**2))) / (
            np.sqrt(2 * np.pi) * delta * T
        )
        rect = np.ones(t_rect.size) / T
        
        # Формирующий импульс
        g_t = np.convolve(h_t, rect) * dt_oversampling

        # Интеграл формирующего импульса
        q_gmsk_oversampling = np.cumsum(g_t) * dt_oversampling

        # Функция S(t), состояющая из 2х частей 
        s_increas = np.sin(np.pi / 2 * q_gmsk_oversampling)
        s_decreas = np.sin(np.pi / 2 - np.pi / 2 * q_gmsk_oversampling)
        s = np.concatenate([s_increas, s_decreas, np.zeros(2)])

        # Формируем основную компоненту разложения Лорана
        # Учитываем, что L - длина ИХ, а не глубина МСИ
        c_0 = np.ones((L + 1) * sps_oversampling)
        for i in range((L + 1) * sps_oversampling):
            for j in range(L):
                c_0[i] *= s[i + j * sps_oversampling]

        # В случае АБГШ c_0 - импульсная характеристика композитного канала
        c_0_trunc = c_0[int(sps_oversampling / 2) : - int(sps_oversampling / 2)]
        h = c_0_trunc[::oversampling]
        E_h = np.sum(np.abs(h)**2)
        h_norm = h / np.sqrt(E_h)

        return h_norm

    def build_reference_burst_waveform(self, burst_active_bits):
            mod = GMSKModulation({
                "BT": self.BT,
                "T": self.T,
                "sps": self.sps,
                "h": self.h,
                "gaus_duration": self.gaus_duration,
                "rect_duration": self.rect_duration,
            })

            tx_ref = mod.process_mod(np.asarray(burst_active_bits, dtype=int))
            return tx_ref
    
    def h_rayleigh(self, rx_burst, tx_ref_burst, debug=False, title_prefix=""):
        sps = self.sps
        train_bit_start = 61
        train_bit_end = 87   # не включая 87, всего 26 бит

        # запас по памяти сигнала для учёта МСИ
        mem_bits = self.L
        # перевод границ из бит в отсчёты с расширением окна
        train_start = (train_bit_start - mem_bits) * sps
        train_end = (train_bit_end + mem_bits) * sps

        # выделение участка тренировочной последовательности 
        rx_train = np.asarray(rx_burst[train_start:train_end], dtype=np.complex128)
        tx_train = np.asarray(tx_ref_burst[train_start:train_end], dtype=np.complex128)

        # оценка delay по корреляции 
        corr = np.correlate(rx_train, tx_train, mode="full") # вычисляет полную взаимную корреляцию входных данных
        delay = np.argmax(np.abs(corr)) - len(tx_train) + 1 

        # Выравнивание 
        if delay > 0:
            rx_train_aligned = rx_train[delay:]
            tx_train_aligned = tx_train[:len(rx_train_aligned)]
        else:
            tx_train_aligned = tx_train[-delay:]
            rx_train_aligned = rx_train[:len(tx_train_aligned)]

        N = min(len(tx_train_aligned), len(rx_train_aligned))

        L_sps = self.est_channel_len_sps

        # y[n] = sum_k h[k] x[n-k] 
        rows = N - L_sps + 1 # количество строк в матрице
        X = np.zeros((rows, L_sps), dtype=np.complex128)

        for i in range(rows):
            seg = tx_train_aligned[i:i + L_sps] # выделение окна в tx_train длиной L_sps
            X[i, :] = seg[::-1] # переворот окна для свертки

        y = rx_train_aligned[L_sps - 1:] 

        reg = self.estimator_reg
        A = X.conj().T @ X + reg * np.eye(L_sps, dtype=np.complex128)
        b = X.conj().T @ y
        h = np.linalg.solve(A, b)

        if debug:

            y_hat_full = np.convolve(tx_train_aligned, h, mode="full")
            y_hat = y_hat_full[:len(rx_train_aligned)]

            fig, ax = plt.subplots(2, 1, figsize=(11, 8))

            ax[0].stem(np.arange(len(h)), np.abs(h))
            ax[0].set_title(f'{title_prefix}Estimated channel impulse response |h_est|')
            ax[0].set_xlabel('Tap index')
            ax[0].set_ylabel('|h_est|')
            ax[0].grid(True)

            ax[1].plot(np.abs(rx_train_aligned), '-', linewidth=1.3, label='|rx_aligned|')
            ax[1].plot(np.abs(y_hat), '--', linewidth=2.0, label='|tx_aligned * h_est|')
            ax[1].set_xlabel('Sample index')
            ax[1].set_ylabel('Magnitude')
            ax[1].legend()
            ax[1].grid(True)

            plt.tight_layout()
            plt.show()

            print("train_start =", train_start, "train_end =", train_end)
            print("len(tx_train_aligned) =", len(tx_train_aligned))
            print("len(rx_train_aligned) =", len(rx_train_aligned))
            print("h_est =", h)


        return h

    def process(self, rx_signal, tx_signal, channel_state = None):
        samples_per_burst = 156 * self.sps
        num_bursts = len(rx_signal) // samples_per_burst
        h_list = []

        use_closed_form_awgn = (
            self.channel_model == "awgn" and not self.force_training_estimation_awgn
        )

        for b in range(num_bursts):
            start_idx = b * samples_per_burst
            end_idx = (b + 1) * samples_per_burst

            rx_burst = rx_signal[start_idx:end_idx]
            tx_burst = tx_signal[start_idx:end_idx]

            if use_closed_form_awgn:
                h_est = self.h_awgn()
            else:
                debug_flag = self.debug_first_burst and (b == 0)

                if self.channel_model == "awgn":
                    h_est = self.h_rayleigh(
                        rx_burst,
                        tx_burst,
                        debug=debug_flag,
                        title_prefix="AWGN: "
                    )
                else:
                    h_est = self.h_rayleigh(
                        rx_burst,
                        tx_burst,
                        debug=debug_flag,
                        title_prefix="Rayleigh: "
                    )

            h_list.append(h_est)

        return h_list