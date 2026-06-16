import numpy as np
import matplotlib.pyplot as plt
from transmitter.modulator import GMSKModulation


GSM_TSC_BITS = {
    1: [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0,
        0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
}

PROFILE_TO_CHANNEL_LEN_SYM = {
    "TU": 2,
    "RA": 1,
    "HT": 6,
}


class ChannelEstimate():

    PLOT_CHANNEL_ESTIMATE = False

    DEFAULTS = {
        "est_channel_len_sym":   None,  # None -> подбирается по channel_profile
        "est_composite_len_sym": 7,     # длина композитной ИХ для composite_derot
        "estimator_reg":         0,  # регуляризация Тихонова
        "tsc_index":             1,     # номер TSC
        "truncate_to_L":         True,  # обрезать композитную ИХ до L*sps
                                        # по окну максимальной энергии
    }

    def __init__(self, modulation_params, simulation_params):
        self.BT = modulation_params.get("BT", 0.3)
        self.T = modulation_params.get("T", 3.69e-6)
        self.sps = modulation_params.get("sps", 4)
        self.gaus_duration = modulation_params.get("gaus_duration", 4)
        self.rect_duration = modulation_params.get("rect_duration", 1)
        # L - длина ИХ передающего фильтра, а не глубина МСИ
        self.L = (self.gaus_duration + self.rect_duration)

        self.channel_model = simulation_params.get("channel_model", "awgn")
        self.channel_profile = simulation_params.get("channel_profile", "TU")
        # Метод оценки канала:
        #   "analytical"      - аналитический h_awgn
        #   "training"        - LS-оценка канала распространения + свёртка с h_awgn
        #                       + обрезка по окну максимальной энергии (опционально)
        #   "composite_derot" - LS-оценка композитной ИХ напрямую через деротацию
        self.estimator_method = simulation_params.get("estimator_method", "composite_derot")

        self.h = modulation_params.get("h", 0.5)

        def _p(key):
            return modulation_params.get(key, self.DEFAULTS[key])

        # Длина оцениваемого канала: либо явно задана, либо по профилю
        explicit_chan_len = _p("est_channel_len_sym")
        if explicit_chan_len is None:
            self.est_channel_len_sym = PROFILE_TO_CHANNEL_LEN_SYM.get(
                self.channel_profile, 2
            )
        else:
            self.est_channel_len_sym = int(explicit_chan_len)

        self.est_composite_len_sym = int(_p("est_composite_len_sym"))
        self.estimator_reg = float(_p("estimator_reg"))
        self.tsc_index = int(_p("tsc_index"))
        self.truncate_to_L = bool(_p("truncate_to_L"))

        # Биты TSC
        tsc_bits_override = modulation_params.get("tsc_bits", None)
        if tsc_bits_override is not None:
            self.tsc_bits = np.asarray(tsc_bits_override, dtype=int)
        else:
            self.tsc_bits = np.asarray(GSM_TSC_BITS[self.tsc_index], dtype=int)

        self._plot_done = False

    # h(t) композитного фильтра (передатчик)
    def h_awgn(self):
        BT = self.BT
        T = self.T
        gaus_duration = self.gaus_duration
        rect_duration = self.rect_duration
        L = self.L

        oversampling = 100
        sps_oversampling = self.sps * oversampling
        dt_oversampling = T / sps_oversampling

        delta = np.sqrt(np.log(2)) / (2 * np.pi * BT)

        t_h = np.arange(-gaus_duration / 2 * T, gaus_duration / 2 * T, dt_oversampling)
        t_rect = np.arange(-rect_duration / 2 * T, rect_duration / 2 * T, dt_oversampling)

        h_t = np.exp(-(t_h**2) / (2 * (delta**2) * (T**2))) / (
            np.sqrt(2 * np.pi) * delta * T
        )
        rect = np.ones(t_rect.size) / T

        g_t = np.convolve(h_t, rect) * dt_oversampling
        q_gmsk_oversampling = np.cumsum(g_t) * dt_oversampling

        s_increas = np.sin(np.pi / 2 * q_gmsk_oversampling)
        s_decreas = np.sin(np.pi / 2 - np.pi / 2 * q_gmsk_oversampling)
        s = np.concatenate([s_increas, s_decreas, np.zeros(2)])

        c_0 = np.ones((L + 1) * sps_oversampling)
        for i in range((L + 1) * sps_oversampling):
            for j in range(L):
                c_0[i] *= s[i + j * sps_oversampling]

        c_0_trunc = c_0[int(sps_oversampling / 2): - int(sps_oversampling / 2)]
        h = c_0_trunc[::oversampling]

        return h

    def build_reference_burst_waveform(self, burst_active_bits):
        mod = GMSKModulation({
            "BT": self.BT, "T": self.T, "sps": self.sps,
            "h": self.h, "gaus_duration": self.gaus_duration,
            "rect_duration": self.rect_duration,
        })
        return mod.process_mod(np.asarray(burst_active_bits, dtype=int))

    def _truncate_by_max_energy(self, h, window_len):

        if len(h) <= window_len:
            return h
        h_abs2 = np.abs(h) ** 2
        # Кумулятивная сумма — быстрее, чем перебор окон в цикле
        cumsum = np.concatenate([[0.0], np.cumsum(h_abs2)])
        best_start = 0
        max_energy = -1.0
        for i in range(len(h) - window_len + 1):
            E = cumsum[i + window_len] - cumsum[i]
            if E > max_energy:
                max_energy = E
                best_start = i
        return h[best_start: best_start + window_len]

    def h_rayleigh(self, rx_burst, tx_ref_burst):
        sps = self.sps
        train_bit_start = 61
        train_bit_end = 87   # 26 бит TSC, не включая 87
        h_awgn = self.h_awgn()

        # Запас по памяти сигнала для учёта МСИ
        mem_bits = self.L - 1
        train_start = (train_bit_start - mem_bits) * sps
        train_end = (train_bit_end + mem_bits) * sps

        rx_train = np.asarray(rx_burst[train_start:train_end], dtype=np.complex128)
        tx_train = np.asarray(tx_ref_burst[train_start:train_end], dtype=np.complex128)

        N = len(tx_train)

        # Длина оцениваемого канала распространения (в отсчётах).
        # Для AWGN можно взять L_sample=1 (канал = дельта-функция).
        if self.channel_model == "awgn":
            L_sample = 1
        else:
            L_sample = self.est_channel_len_sym * sps

        # LS: rx_train[n] = sum_k h_chan[k] * tx_train[n-k] + w
        rows = N - L_sample + 1
        X = np.zeros((rows, L_sample), dtype=np.complex128)
        for i in range(rows):
            seg = tx_train[i:i + L_sample]
            X[i, :] = seg[::-1]
        y = rx_train[L_sample - 1:]

        reg = self.estimator_reg
        A = X.conj().T @ X + reg * np.eye(L_sample, dtype=np.complex128)
        b = X.conj().T @ y
        h_chan = np.linalg.solve(A, b)

        # Композитная ИХ = передатчик ⊛ канал распространения
        h = np.convolve(h_awgn, h_chan)

        # Обрезка по окну максимальной энергии до L*sps отсчётов.
        # Это убирает шумовые хвосты и согласует длину с памятью MLSE.
        if self.truncate_to_L:
            h = self._truncate_by_max_energy(h, self.L * sps)

        return h_chan, h, rx_train, tx_train

    def h_composite_derot(self, rx_burst):
        sps = self.sps
        train_bit_start = 61
        train_bit_end = 87

        L_comp = self.est_composite_len_sym * sps

        train_start = train_bit_start * sps
        train_end = train_bit_end * sps
        rx_train = np.asarray(rx_burst[train_start:train_end], dtype=np.complex128)

        # Деротация на отсчётной сетке
        n_local = np.arange(len(rx_train))
        derot = np.exp(-1j * np.pi * (n_local + sps) / (2 * sps))
        rx_derot = rx_train * derot

        # Псевдо-символы Лорана (вывод через дифф. предкодирование GSM)
        alpha = 1.0 - 2.0 * self.tsc_bits.astype(float)

        n_tsc = len(alpha)
        a_up = np.zeros(n_tsc * sps, dtype=np.complex128)
        a_up[::sps] = alpha

        N = len(rx_derot)
        rows = N - L_comp + 1
        if rows < L_comp:
            raise ValueError(
                f"TSC слишком короткая для оценки композита длиной {L_comp} отсчётов."
            )

        X = np.zeros((rows, L_comp), dtype=np.complex128)
        for i in range(rows):
            X[i, :] = a_up[i:i + L_comp][::-1]
        y = rx_derot[L_comp - 1:]

        reg = self.estimator_reg
        A = X.conj().T @ X + reg * np.eye(L_comp, dtype=np.complex128)
        b = X.conj().T @ y
        h_comp_derot = np.linalg.solve(A, b)

        # Ротация обратно
        k = np.arange(L_comp)
        rot_back = np.exp(1j * np.pi * k / (2 * sps))
        h_comp = h_comp_derot * rot_back

        # Фазовая компенсация по главному пику: убирает глобальный поворот
        # оценки, который смешивает Re/Im компоненты на выходе matched
        # filter и ломает деротацию в детекторе.
        peak_idx = int(np.argmax(np.abs(h_comp)))
        h_comp = h_comp * np.exp(-1j * np.angle(h_comp[peak_idx]))

        # Обрезка по окну максимальной энергии (для согласованности с MLSE)
        if self.truncate_to_L:
            h_comp = self._truncate_by_max_energy(h_comp, self.L * sps)

        return h_comp, h_comp_derot, rx_train, rx_derot

    def _plot_estimate(self, h_chan, h, rx_train_aligned, tx_train_aligned):
        title_prefix = f"{self.channel_model} | {self.estimator_method}: "

        fig, ax = plt.subplots(2, 1, figsize=(11, 8))

        ax[0].stem(np.arange(len(h)), np.abs(h))
        ax[0].set_title(f'{title_prefix}Composite IR |h|')
        ax[0].set_xlabel('Tap index')
        ax[0].set_ylabel('|h|')
        ax[0].grid(True)

        ax[1].stem(np.arange(len(h_chan)), np.abs(h_chan))
        ax[1].set_title(f'{title_prefix}Channel IR |h_chan|')
        ax[1].set_xlabel('Tap index')
        ax[1].set_ylabel('|h_chan|')
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    def _plot_composite(self, h_comp, h_comp_derot):
        title_prefix = f"{self.channel_model} | {self.estimator_method}: "

        fig, ax = plt.subplots(3, 1, figsize=(11, 10))

        ax[0].stem(np.arange(len(h_comp)), np.abs(h_comp))
        ax[0].set_title(f'{title_prefix}Composite IR |h_comp|')
        ax[0].grid(True)

        ax[1].plot(h_comp_derot.real, 'o-', label='Re')
        ax[1].plot(h_comp_derot.imag, 'x-', label='Im')
        ax[1].set_title(f'{title_prefix}Derotated composite IR')
        ax[1].legend(); ax[1].grid(True)

        h_ref = self.h_awgn()
        ax[2].stem(np.arange(len(h_ref)), np.abs(h_ref))
        ax[2].set_title('Reference |h_awgn| (transmitter only)')
        ax[2].grid(True)

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
                    self._plot_estimate(h_chan, h_est, rx_aligned, tx_aligned)
                    self._plot_done = True

            elif self.estimator_method == "composite_derot":
                h_est, h_comp_derot, rx_aligned, rx_derot = self.h_composite_derot(rx_burst)
                if self.PLOT_CHANNEL_ESTIMATE and not self._plot_done and b == 0:
                    self._plot_composite(h_est, h_comp_derot)
                    self._plot_done = True

            else:
                raise ValueError(
                    f"Unknown estimator_method: {self.estimator_method!r}. "
                    f"Expected 'analytical', 'training', or 'composite_derot'."
                )

            h_list.append(h_est)

        return h_list
