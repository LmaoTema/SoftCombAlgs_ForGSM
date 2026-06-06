import numpy as np
import matplotlib.pyplot as plt
from transmitter.modulator import GMSKModulation
class ChannelEstimate():
    
    # Если True, на первом бёрсте всего прогона строится диагностика оценки канала:
    PLOT_CHANNEL_ESTIMATE = False

    DEFAULTS = {
        "est_channel_len_sym": 4,      # длина оцениваемой ИХ в отсчётах (None -> 5*sps)
        "estimator_reg":       1e-4,   # регуляризация Тихонова в LS-оценке (X^H X + reg*I)
    }

    def __init__(self, modulation_params, simulation_params):
        self.BT = modulation_params.get("BT", 0.3)
        self.T = modulation_params.get("T", 3.69e-6)
        self.sps = modulation_params.get("sps", 4)
        self.gaus_duration = modulation_params.get("gaus_duration", 4)
        self.rect_duration = modulation_params.get("rect_duration", 1)
        # L - длина ИХ, а не глубина МСИ
        self.L = (self.gaus_duration + self.rect_duration)

        self.channel_model = simulation_params.get("channel_model", "awgn")
        # Метод оценки канала:
        #   "analytical" - h_awgn 
        #   "training"   - по тренировочной последовательности 
        self.estimator_method = simulation_params.get("estimator_method", "training")

        self.h = modulation_params.get("h", 0.5)

        def _p(key):
            return modulation_params.get(key, self.DEFAULTS[key])

        _len = _p("est_channel_len_sym")
        self.est_channel_len_sym = int(_p("est_channel_len_sym"))
        self.estimator_reg = float(_p("estimator_reg"))

        # Флаг, чтобы график рисовался один раз за весь прогон, а не на каждой точке.
        self._plot_done = False
    
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

        return h

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
        
    def _best_decimation_phase(self, rx_train, tx_train):
        
        sps = self.sps
        best_phase, best_score = 0, -1.0
        for ph in range(sps):
            txs = tx_train[ph::sps]
            rxs = rx_train[ph::sps]
            n = min(len(txs), len(rxs))
            if n < 2:
                continue
            score = np.abs(np.vdot(txs[:n], rxs[:n]))
            if score > best_score:
                best_score, best_phase = score, ph
        return best_phase
    
    def h_rayleigh(self, rx_burst, tx_ref_burst):
        sps = self.sps
        train_bit_start = 61
        train_bit_end = 87   # не включая 87, всего 26 бит
        h_awgn = self.h_awgn()
        # запас по памяти сигнала для учёта МСИ
        mem_bits = self.L
        # перевод границ из бит в отсчёты с расширением окна
        train_start = (train_bit_start - mem_bits) * sps
        train_end = (train_bit_end + mem_bits) * sps

        # выделение участка тренировочной последовательности 
        rx_train = np.asarray(rx_burst[train_start:train_end], dtype=np.complex128)
        tx_train = np.asarray(tx_ref_burst[train_start:train_end], dtype=np.complex128)

        phase = self._best_decimation_phase(rx_train, tx_train)
        tx_sym = tx_train[phase::sps]
        rx_sym = rx_train[phase::sps]
        
        # оценка delay по корреляции 
        corr = np.correlate(rx_sym, tx_sym, mode="full") # вычисляет полную взаимную корреляцию входных данных
        delay = np.argmax(np.abs(corr)) - len(tx_sym) + 1 

        # Выравнивание 
        if delay > 0:
            rx_sym = rx_sym[delay:]
            tx_sym = tx_sym[:len(rx_sym)]
        else:
            tx_sym = tx_sym[-delay:]
            rx_sym = rx_sym[:len(tx_sym)]

        N = min(len(tx_sym), len(rx_sym))

        L_sym = self.est_channel_len_sym

        # y[n] = sum_k h[k] x[n-k] 
        rows = N - L_sym + 1 # количество строк в матрице
        X = np.zeros((rows, L_sym), dtype=np.complex128)

        for i in range(rows):
            seg = tx_sym[i:i + L_sym] # выделение окна в tx_train длиной L_sps
            X[i, :] = seg[::-1] # переворот окна для свертки

        y = rx_sym[L_sym - 1:] 

        reg = self.estimator_reg
        A = X.conj().T @ X + reg * np.eye(L_sym, dtype=np.complex128)
        b = X.conj().T @ y
        h_chan = np.linalg.solve(A, b)

        h = np.convolve(h_awgn, h_chan)

        return h_chan, h, rx_sym, tx_sym

    def _plot_estimate(self, h_chan, h, rx_train_aligned, tx_train_aligned):
        
        y_hat_full = np.convolve(tx_train_aligned, h, mode="full")
        y_hat = y_hat_full[:len(rx_train_aligned)]
        title_prefix = f"{self.channel_model} | {self.estimator_method}: "

        fig, ax = plt.subplots(2, 1, figsize=(11, 8))

        ax[0].stem(np.arange(len(h)), np.abs(h))
        ax[0].set_title(f'{title_prefix}Compozite channel impulse response |h_est|')
        ax[0].set_xlabel('Tap index')
        ax[0].set_ylabel('|h|')
        ax[0].grid(True)

        ax[1].stem(np.arange(len(h_chan)), np.abs(h_chan))
        ax[1].set_title(f'{title_prefix}Estimated channel impulse response |h_est|')
        ax[1].set_xlabel('Tap index')
        ax[1].set_ylabel('|h_est|')
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    def process(self, rx_signal, tx_signal):
        samples_per_burst = 156 * self.sps
        num_bursts = len(rx_signal) // samples_per_burst
        h_list = []

        for b in range(num_bursts):
            start_idx = b * samples_per_burst
            end_idx = (b + 1) * samples_per_burst

            rx_burst = rx_signal[start_idx:end_idx]
            tx_burst = tx_signal[start_idx:end_idx]

            if self.estimator_method == "analytical":

                h_est = self.h_awgn()
            elif self.estimator_method == "training":

                h_chan, h_est, rx_aligned, tx_aligned = self.h_rayleigh(rx_burst, tx_burst)

                if self.PLOT_CHANNEL_ESTIMATE and not self._plot_done and b == 0:
                    self._plot_estimate(h_chan, h_est,  rx_aligned, tx_aligned)
                    self._plot_done = True
            else:
                raise ValueError(
                    f"Unknown estimator_method: {self.estimator_method!r}. "
                    f"Expected 'analytical' or 'training'."
                )

            h_list.append(h_est)

        return h_list