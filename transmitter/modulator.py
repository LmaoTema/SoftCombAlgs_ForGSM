import numpy as np
from core.block import Block
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
        self.h = params.get("h", 0.5)
        self.gaus_duration = params.get("gaus_duration", 4)
        self.rect_duration = params.get("rect_duration", 1)

        return

    def differential_encoding(self, bits):
        if bits.size % 148 != 0:
            raise ValueError("Количество модуляционных бит должно быть кратным 148")

        # Создаем сдвинутую на 1 отсчет последовательность бит 
        bits_previous = np.zeros(bits.size, dtype=int)
        bits_previous[1:] = bits[:-1]

        # Диф.кодирование
        d_curr = bits ^ bits_previous
        alpha = 1 - 2 * d_curr

        # alpha = 1 - 2 * bits

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

        # Формируем гауссовский и прямоугольный импульсы
        h_t = np.exp(-(t_h**2) / (2 * (delta**2) * (T**2))) / (
            np.sqrt(2 * np.pi) * delta * T
        )
        rect = np.ones(t_rect.size) / T

        # Формирующий импульс
        g_t = np.convolve(h_t, rect) * dt_oversampling

        # Интеграл формирующего импульса
        q_gmsk_oversampling = np.cumsum(g_t) * dt_oversampling
        q_gmsk = q_gmsk_oversampling[::oversampling]
        
        return q_gmsk

    def calc_phase(self, alpha, q_gmsk):
        h = self.h
        sps = self.sps
        gaus_duration = self.gaus_duration
        rect_duration = self.rect_duration

        num_bits = alpha.size
        phi = np.zeros(num_bits * sps + q_gmsk.size - sps)

        is_plot_phase = False

        if is_plot_phase:
            phi_plot = np.zeros(20)
            shift_plot = 2
            step = 0

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)       
            ax1.step(np.arange(10 - shift_plot), alpha[:10 - shift_plot], where='post')
            ax1.set_ylabel("Symbols")
            ax1.set_xlabel("t / T") 
            ax1.grid()
            ax1.set_xlim(-0.5, 10.5 - shift_plot)
            

        for i in range(num_bits):
            alpha_i = alpha[i]
            start_idx = i * sps

            phi[start_idx : start_idx + q_gmsk.size] += alpha_i * np.pi * h * q_gmsk
            
            if is_plot_phase:
                if i < 6:
                    phi_plot = alpha_i * np.pi * h * q_gmsk
                    ax2.plot((np.arange(20) + step) / 4 - shift_plot, phi_plot, lw=3)
                    step += 4

            # Прибавляем итоговое накопленное значение ко всему массиву
            phase_step = alpha_i * np.pi * h
            phi[start_idx + q_gmsk.size :] += phase_step

        # Убираем задержку на 2T, вносимую гауссовским филльтром
        # Сдвиг на 1 отсчет, чтобы пик текущего символа был в конце соответствующего символьного интервала 
        shift = (gaus_duration + rect_duration) / 2 - 0.5

        phi_shift = phi[int(shift * sps) + 1 : - int(shift * sps) + 1]

        phi_shift = phi[int(shift * sps) : - int(shift * sps)]

        if is_plot_phase:

            ax2.set_ylabel("Phase step")
            ax2.set_xlabel("t / T") 
            ax2.grid()
            ax2.set_xlim(-0.5, 10.5 - shift_plot)

            ax3.plot(np.arange((10 - shift_plot) * 4) / 4, phi[shift_plot * 4 : 40], lw=3)
            ax3.axhline(y=np.pi, color='r', linestyle='--', lw=1)
            ax3.axhline(y=np.pi * 3 / 2, color='r', linestyle='--', lw=1)
            ax3.set_ylabel("Phase") 
            ax3.set_xlabel("t / T") 
            ax3.grid()
            ax3.set_xlim(-0.5, 10.5 - shift_plot)

            plt.suptitle("Со сдвигом") 
            plt.tight_layout()
            plt.show()

        return phi_shift

    def h_mod(self):
        BT = self.BT
        T = self.T
        gaus_duration = self.gaus_duration
        rect_duration = self.rect_duration
        L = gaus_duration + rect_duration

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
    
    def liner_mod(self, bits):

        num_bits = len(bits)
        sps = self.sps

        # Символы и ИХ
        alpha = self.differential_encoding(bits)
        phase_accum = np.cumsum(alpha) * (np.pi / 2)
        a_n = np.exp(1j * phase_accum)
        h = self.h_mod()

        # Инициализируем
        sig_len = num_bits * sps + h.size
        linear_signal  = np.zeros(sig_len, dtype=complex)

        is_plot_linear = False

        if is_plot_linear:
            lin_sig_plot = np.zeros(20)
            shift_plot = 6
            step = 0

            y_size = 10
            x_size = 10
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        
        # линейная модуляция
        for i in range(num_bits):
            start = i * sps
            # end = start + T + 4T - мси
            end = start + h.size
            linear_signal[start:end] += a_n[i] * h
            if i < 8:
                if is_plot_linear:
                    lin_sig_plot = a_n[i] * h

                    ax2.plot((np.arange(20) + step - shift_plot) / 4, np.real(lin_sig_plot), lw=3)
                    
                    ax3.plot((np.arange(20) + step - shift_plot) / 4, np.imag(lin_sig_plot), lw=3)
                    
                    step += 4

        if is_plot_linear:
            ax1.step(np.arange(11), alpha[:11], where='post', lw=3)
            ax1.set_ylabel("Symbols", fontsize=y_size)
            ax1.set_xlabel("t / T", fontsize=x_size)
            ax1.grid()
            ax1.set_xlim(-0.5, 10.5)

            ax2.set_ylabel("Re", fontsize=y_size)
            ax2.set_xlabel("t / T", fontsize=x_size)
            ax2.grid()
            ax2.set_xlim(-0.5, 10.5)
            
            ax3.set_ylabel("Im", fontsize=y_size)
            ax3.set_xlabel("t / T", fontsize=x_size)
            ax3.grid()
            ax3.set_xlim(-0.5, 10.5)
            
            plt.tight_layout()
            plt.show()

        # Сдвиг для совмещения: отрезаем 7, чтобы пик был на индексе 3 (конец символьного интервала)
        # (10 - 7 = 3)
        linear_final = linear_signal[6 : num_bits * sps + 6]

        return linear_final
    
    def process_mod(self, bits):
        
        # Делим на 148, а не 156, что бы без кодера тоже работало
        # Так как берем целую часть, то на результат не влияет 
        active_size = 148
        num_bursts = len(bits) // active_size
    
        q_gmsk = self.generate_q_gmsk()
        
        all_signals = []
        all_linear_signals = []
        guard_period = np.zeros(8 * self.sps, dtype=complex)
        gp_len = 0
        for i in range(num_bursts):
            # Вырезаем из потока интересующий пакет (без защитного интервала)
            burst = bits[gp_len + i*active_size : gp_len + (i+1)*active_size]
            gp_len += 8

            # Модуляция
            alpha = self.differential_encoding(burst)
            phi = self.calc_phase(alpha, q_gmsk)
            signal_envelope = np.exp(1j * phi)

            signal_linear_envelope = self.liner_mod(burst)

            is_plot = False
            
            if is_plot:
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,6))
                # fig, (ax1, ax2) = plt.subplots(2, 1)
                y_size = 16
                x_size = 14

                ax1.step(np.arange(148), alpha[:148], where="post", lw=3)
                ax1.set_ylabel("Символы", fontsize=y_size)
                ax1.set_xlabel("t / T", fontsize=x_size)
                ax1.grid()
                # ax1.set_xlim(-0.5, 10.5)
                # ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))

                ax2.plot(np.arange(148*4) / 4, phi[:148*4], lw=3)
                ax2.set_ylabel("Фаза", fontsize=y_size)
                ax2.set_xlabel("t / T", fontsize=x_size)
                ax2.grid(True)
                # ax2.set_xlim(-0.5, 10.5) 

                # --- НАСТРОЙКА ОСИ С PI ---
                # 1. Устанавливаем шаг сетки каждые pi/2
                ax2.yaxis.set_major_locator(ticker.MultipleLocator(2 * np.pi))
                # ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))

                # 2. Функция для создания красивых подписей
                def format_func(value, f):
                    n = round(2 * value / np.pi)
                    if n == 0:
                        return "0"
                    elif n == 1:
                        return "$pi/2$"
                    elif n == -1:
                        return "$-pi/2$"
                    elif n % 2 == 0:
                        res = n // 2
                        if res == 1: 
                            return "$pi$"
                        if res == -1:
                            return "$-pi$"
                        
                        return f"${res}pi$"
                    else:
                        return f"${n}pi/2$"

                ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
                # Увеличиваем размер шрифта цифр (меток) на осях
                # ax2.tick_params(axis='both', which='major', labelsize=12)


                ax3.plot(np.arange(148*4) / 4, np.real(signal_envelope[:148*4]), lw=3)
                ax3.set_ylabel("Вещественная часть", fontsize=y_size)
                ax3.set_xlabel("t / T", fontsize=x_size)
                ax3.grid()
                # ax3.set_xlim(-0.5, 10.5)
                # ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))

                ax4.plot(np.arange(148*4) / 4, np.imag(signal_envelope[:148*4]), lw=3)
                ax4.set_ylabel("Мнимая часть", fontsize=y_size)
                ax4.set_xlabel("t / T", fontsize=x_size)
                ax4.grid()
                # ax4.set_xlim(-0.5, 10.5)
                # ax4.xaxis.set_major_locator(ticker.MultipleLocator(1))

                plt.tight_layout()
                plt.show()
            
            # Добавляем сигнал и защитный интервал
            all_signals.append(signal_envelope)
            all_signals.append(guard_period)

            all_linear_signals.append(signal_linear_envelope)
            all_linear_signals.append(guard_period)

        return np.concatenate(all_signals), np.concatenate(all_linear_signals)


class PSKModulation:
    def __init__(self, params):
        raise ValueError("Еще не реализован")
