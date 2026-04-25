import numpy as np
import matplotlib.pyplot as plt

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

    #def h_rayleigh(self, rx_burst, tx_burst):
        L_sps = self.L * self.sps
        train_start = 61 * self.sps
        train_end = 87 * self.sps

        tx_train = tx_burst[train_start: train_end]
        rx_train = rx_burst[train_start : train_end]

        N = len(tx_train)
        X = np.zeros((N - L_sps, L_sps), dtype=complex)

        for i in range(N - L_sps):
            X[i] = tx_train[i:i + L_sps]

        y = rx_train[L_sps:]

        h = np.linalg.lstsq(X, y, rcond=None)[0]

        return h
    
    def h_rayleigh(self, rx_burst, tx_burst):
        sps = self.sps
        L_sps = 2 * sps   

        train_start = 61 * sps
        train_end = 87 * sps

        tx_train = tx_burst[train_start:train_end]
        rx_train = rx_burst[train_start:train_end]

        corr = np.correlate(rx_train, tx_train, mode='full')
        delay = np.argmax(np.abs(corr)) - len(tx_train) + 1

        if delay > 0:
            rx_train = rx_train[delay:]
        else:
            tx_train = tx_train[-delay:]

        min_len = min(len(tx_train), len(rx_train))
        tx_train = tx_train[:min_len]
        rx_train = rx_train[:min_len]


        #plt.figure()
        #plt.plot(np.real(tx_train), label="TX")
        #plt.plot(np.real(rx_train), label="RX")
        #plt.legend()
        #plt.title("TX vs RX (aligned)")
        #plt.grid()
        #plt.show()
        
        N = len(tx_train)

        X = np.zeros((N - L_sps + 1, L_sps), dtype=complex)
        for i in range(N - L_sps + 1):
            X[i] = tx_train[i:i + L_sps]

        y = rx_train[L_sps - 1:]

        h = np.linalg.inv(X.conj().T @ X + 1e-1 * np.eye(L_sps)) @ X.conj().T @ y

        #plt.stem(np.abs(h))
        #plt.title("Estimated channel impulse response")
        #plt.show()
        
        return h

    def process(self, rx_signal, tx_signal):
        
        if self.channel_model == "awgn":
            h_est = self.h_awgn()
       
        samples_per_burst = 156 * self.sps
        num_bursts = len(rx_signal) // samples_per_burst

        h_list = []

        for b in range(num_bursts):
            start_idx = b * samples_per_burst
            end_idx = (b + 1) * samples_per_burst

            rx_burst = rx_signal[start_idx:end_idx]
            tx_burst = tx_signal[start_idx:end_idx]

            if self.channel_model != "awgn":
                h_est = self.h_rayleigh(rx_burst, tx_burst)

            h_list.append(h_est)

        return h_list

            
