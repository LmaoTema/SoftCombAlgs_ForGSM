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
        # Определяем начало и конец TSC
        train_start = (61) * sps
        train_end = (87) * sps

        # Выделяем последовательности
        tx_train = tx_burst[train_start:train_end]
        rx_train = rx_burst[train_start:train_end]

        # Синхронизируемся на центр TSC (По идее должны остаться там же, так как у нас идеальная синхронизация, но Василий сказал сделать)
        corr_train = np.convolve(tx_train, np.conj(rx_train[::-1]))
        peak_idx = np.argmax(np.abs(corr_train))

        # Разворачиваем фазу сигнала обратно
        phase_correction = np.conj(corr_train[peak_idx]) / np.abs(corr_train[peak_idx])
        rx_burst_aligned = (rx_burst * phase_correction)

        # Сдвигаем сигнал к макисмуму в 3 остчет
        shift = 103 - peak_idx
        zeros = np.zeros(np.abs(shift), dtype=complex)
        if shift > 0:
            rx_burst_aligned_shift = np.concatenate((zeros, rx_burst_aligned[:-shift]))
        elif shift < 0:
            rx_burst_aligned_shift = np.concatenate((rx_burst_aligned[np.abs(shift):],zeros))
        else:
            rx_burst_aligned_shift = rx_burst_aligned

        # Длина ИХ канала (в отсчетах)
        if self.channel_model == "awgn":
            h_len = 1
        else:
            if self.channel_profile == "TU":
                h_len = 2 * sps
            elif self.channel_profile == "RA":
                raise ValueError("Оценка Андрюхи пока что толкьо для TU")
            elif self.channel_profile == "HT":
                raise ValueError("Оценка Андрюхи пока что толкьо для TU")
            else:
                raise ValueError("Оценка Андрюхи пока что толкьо для TU")

        # Задаем основные переменные
        start_train_idx = 3 * sps 
        train_len = 26 * sps
        num_rows = train_len - h_len - start_train_idx + 1

        # Выерзаем из тренировочных бит сдвинутого сигнала используемые в матрице отсчеты
        rx_train_aligned = rx_burst_aligned_shift[train_start : train_end]
        rx_train_cut = rx_train_aligned[h_len + start_train_idx - 1 :]

        # Определим матрицу свертки
        conv_matrix = np.zeros((num_rows, h_len), dtype=complex)
        for i in range(num_rows):
            row = tx_train[start_train_idx + i : h_len + start_train_idx + i]
            conv_matrix[i, :] = row[::-1]

        # Ищем коэффициенты канала по формуле
        inv_abs_matrix = np.linalg.inv(conv_matrix.conj().T @ conv_matrix)
        h_chan = inv_abs_matrix @ conv_matrix.conj().T @ rx_train_cut

        # Получаем коэффициенты композитного канала
        h_awgn = self.h_awgn()
        h = np.convolve(h_awgn, h_chan)

        # Берем из h окно длиной 5 символов с максимальной энергией
        len_window = 5
        start_idx_max_energy = 0
        max_energy = 0
        for i in range(len(h) - len_window * sps + 1):
            window = h[i : i + len_window * sps]
            E = np.sum(np.abs(window)**2)
            if E > max_energy:
                max_energy = E
                start_idx_max_energy = i

        h_truncated = h[start_idx_max_energy : start_idx_max_energy + 5 * sps]

        return rx_burst_aligned_shift, h_truncated

    def process(self, rx_signal, tx_signal):
        samples_per_burst = 156 * self.sps
        num_bursts = len(rx_signal) // samples_per_burst
        h_list = []
        rx_burst_aligned_all = []

        for b in range(num_bursts):
            start_idx = b * samples_per_burst
            end_idx = (b + 1) * samples_per_burst

            rx_burst = rx_signal[start_idx:end_idx]
            tx_burst = tx_signal[start_idx:end_idx]

            if self.estimator_method == "analytical":
                h_est = self.h_awgn()
                rx_burst_aligned = rx_signal
            elif self.estimator_method == "training":
                rx_burst_aligned, h_est  = self.h_rayleigh(rx_burst, tx_burst)

            h_list.append(h_est)
            rx_burst_aligned_all.append(rx_burst_aligned)

        return np.concatenate(rx_burst_aligned_all), h_list