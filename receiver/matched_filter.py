import numpy as np
from core.block import Block
import matplotlib.pyplot as plt


class MatchedFilter(Block):

    def __init__(self, modulation_params, is_working=False):
        super().__init__(is_working)

        self.sps = modulation_params.get("sps", 4)

    def _process(self, rx_signal, h):
        
        samples_per_burst = 156 * self.sps
        num_bursts = len(rx_signal) // samples_per_burst

        match_signal = []

        for b in range(num_bursts):
            # Разные оценки h для разных пакетов
            h_est = h[b]
            start_idx = b * samples_per_burst
            end_idx = (b + 1) * samples_per_burst
    
            rx_burst = rx_signal[start_idx  : end_idx]

            # Свертка сигнала с ИХ СФ
            E_h = np.sum(np.abs(h_est)**2)
            h_mf = np.conj(h_est[::-1]) / np.sqrt(E_h)

            is_plot_h = False

            if is_plot_h:
                fig, (ax1) = plt.subplots(1,1)

                ax1.plot(np.arange(20), h_est, lw=3, label="h")
                ax1.plot(np.arange(20), h_mf, lw=3, label="$h_{mf}$")
                ax1.set_ylabel("A")
                ax1.set_xlabel("t / T")
                ax1.legend()

                plt.show()

            burst_match = np.convolve(rx_burst, h_mf)
            # Поиск главного пика
            peak_idx = int(np.argmax(np.abs(h_est)))
            # Считаем сдвиг после свертки
            # delay = len(h_est) - 1 - peak_idx
            delay = len(h_est) - 1 - peak_idx
            # Убираем задержку от СФ.
            burst_trunc = burst_match[delay: delay + len(rx_burst)]
            # Склеиваем все пакеты
            match_signal.append(burst_trunc)

            is_plot_mf = False

            if is_plot_mf:
                fig, (ax1) = plt.subplots(1,1)

                ax1.plot(np.arange(40), rx_burst[:40], lw=1, label="rx")
                shift = 7
                ax1.plot(np.arange(40), burst_match[shift: shift + 40], lw=2, label=f"shift = {shift}")
                shift = 8
                ax1.plot(np.arange(40), burst_match[shift: shift + 40], lw=2, label=f"shift = {shift}")
                shift = 9
                ax1.plot(np.arange(40), burst_match[shift: shift + 40], lw=2, label=f"shift = {shift}")
                shift = 10
                ax1.plot(np.arange(40), burst_match[shift: shift + 40], lw=2, label=f"shift = {shift}")
                ax1.set_ylabel("A")
                ax1.set_xlabel("t / T")
                ax1.grid()
                ax1.legend()

                plt.show()

        return np.concatenate(match_signal)
