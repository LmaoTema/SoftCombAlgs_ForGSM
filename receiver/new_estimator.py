import numpy as np
import matplotlib.pyplot as plt
from transmitter.modulator import GMSKModulation
class ChannelEstimate():

    def __init__(self, modulation_params, simulation_params):
        self.BT = modulation_params.get("BT", 0.3)
        self.T = modulation_params.get("T", 3.69e-6)
        self.sps = modulation_params.get("sps", 4)
        self.h = modulation_params.get("h", 0.5)
        self.gaus_duration = modulation_params.get("gaus_duration", 4)
        self.rect_duration = modulation_params.get("rect_duration", 1)
        # L - длина ИХ передатчика, а не глубина МСИ
        self.L = (self.gaus_duration + self.rect_duration)
        
        self.channel_model = simulation_params.get("channel_model", "awgn")
        self.channel_profile = simulation_params.get("channel_profile", "TU")
        self.estimator_method = simulation_params.get("estimator_method", "training")
        self.estimator_reg = 1e-4
    
    # Аналитическая h(t) композитного фильтра (передатчик + канал)
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
    
    def h_rayleigh(self, rx_burst, tx_burst):
        sps = self.sps
        train_bit_start = 61
        train_bit_end = 87

        # запас по памяти сигнала для учёта МСИ
        mem_bits = self.L - 1
        # перевод границ из бит в отсчёты с расширением окна
        train_start = (train_bit_start - mem_bits) * sps
        train_end = (train_bit_end + mem_bits) * sps
        
        # выделение участка тренировочной последовательности 
        rx_train = np.asarray(rx_burst[train_start:train_end], dtype=np.complex128)
        tx_train = np.asarray(tx_burst[train_start:train_end], dtype=np.complex128)

        # Длина ИХ канала (или сразу берестся для композитного канала?) (в отсчетах)
        if self.channel_model == "awgn":
            L_sample = 1
        else:
            if self.channel_profile == "TU":
                L_sample = 2 * sps
            elif self.channel_profile == "RA":
                raise ValueError("Оценка Андрюхи пока что толкьо для TU")
            elif self.channel_profile == "HT":
                raise ValueError("Оценка Андрюхи пока что толкьо для TU")
            else:
                raise ValueError("Оценка Андрюхи пока что толкьо для TU")

        # Длина тренировочной последовательности (в отсчетах)
        N = len(rx_train)
        # Сдвиг в тренировочной последовательности на длину ИХ композитного канала
        rows = N - L_sample + 1 # количество строк в матрице
        X = np.zeros((rows, L_sample), dtype=np.complex128)

        for i in range(rows):
            seg = tx_train[i:i + L_sample] # выделение окна в tx_train длиной L_sps
            X[i, :] = seg[::-1] # переворот окна для свертки

        y = rx_train[L_sample - 1:] 

        # Матрично находим коэффициенты канала
        reg = self.estimator_reg
        A = X.conj().T @ X + reg * np.eye(L_sample, dtype=np.complex128)
        b = X.conj().T @ y
        h_chan = np.linalg.solve(A, b)

        h_awgn = self.h_awgn()

        h = np.convolve(h_awgn, h_chan)

        # Ищем окно L = 5 с максимальной энергией
        start_idx_max_energy = 0
        max_energy = 0
        for i in range(len(h) - self.L * sps + 1):
            window = h[i : i + self.L * sps]
            E = np.sum(np.abs(window)**2)
            if E > max_energy:
                max_energy = E
                start_idx_max_energy = i

        h_truncated = h[start_idx_max_energy : start_idx_max_energy + self.L * sps]

        return h_truncated

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
                h_est = self.h_rayleigh(rx_burst, tx_burst)

            h_list.append(h_est)

        return h_list