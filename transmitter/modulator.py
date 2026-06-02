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
            ax_size = 14

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,6))       
            ax1.step(np.arange(25 - shift_plot), alpha[:25 - shift_plot], where='post', lw=4)
            ax1.set_ylabel("Символы", fontsize=ax_size)
            ax1.set_xlabel("t / T", fontsize=ax_size) 
            ax1.grid()
            ax1.set_xlim(-0.5, 10.5)
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
            

        for i in range(num_bits):
            alpha_i = alpha[i]
            start_idx = i * sps

            phi[start_idx : start_idx + q_gmsk.size] += alpha_i * np.pi * h * q_gmsk
            
            if is_plot_phase:
                if i < 12:
                    phi_plot = alpha_i * np.pi * h * q_gmsk
                    ax2.plot((np.arange(20) + step) / 4 - shift_plot, phi_plot, lw=4)
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
                    

            ax2.set_ylabel("Вклады симв., рад", fontsize=ax_size)
            ax2.set_xlabel("t / T", fontsize=ax_size) 
            ax2.grid()
            ax2.set_xlim(-0.5, 10.5)
            # ax2.set_ylim(0, 7)
            ax2.yaxis.set_major_locator(ticker.MultipleLocator(np.pi/2))
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))

            ax3.plot(np.arange((25 - shift_plot) * 4) / 4, phi[shift_plot * 4 : 100], lw=4)
            ax3.axhline(y=np.pi, color='r', linestyle='--', lw=1)
            ax3.axhline(y=np.pi * 3 / 2, color='r', linestyle='--', lw=1)
            ax3.axhline(y=3*np.pi, color='r', linestyle='--', lw=1)
            ax3.axhline(y=np.pi * 7 / 2, color='r', linestyle='--', lw=1)
            ax3.set_ylabel("Рез. фаза, рад", fontsize=ax_size) 
            ax3.set_xlabel("t / T", fontsize=ax_size) 
            ax3.set_ylim(0, 12)
            ax3.grid()
            ax3.set_xlim(-0.5, 10.5)
            ax3.yaxis.set_major_locator(ticker.MultipleLocator(np.pi/2))
            ax3.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
            ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))

            # plt.suptitle("Со сдвигом") 
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

        is_plot_h = False

        if is_plot_h:
            ax_size = 16

            plt.figure(figsize=(12,6))
            plt.plot(np.arange(len(c_0)) / 400, c_0, lw=4, c="r")
            plt.grid()
            plt.xlabel("t / T", fontsize=ax_size)
            plt.ylabel("$c_0(t)$", fontsize=ax_size)
            plt.show()

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

            y_size = 14
            x_size = 14
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))
        
        # линейная модуляция
        for i in range(num_bits):
            start = i * sps
            # end = start + T + 4T - мси
            end = start + h.size
            linear_signal[start:end] += a_n[i] * h
            if i < 10:
                if is_plot_linear:
                    lin_sig_plot = a_n[i] * h

                    ax1.plot((np.arange(20) + step - shift_plot) / 4, np.real(lin_sig_plot), lw=4)
                    
                    ax2.plot((np.arange(20) + step - shift_plot) / 4, np.imag(lin_sig_plot), lw=4)
                    
                    step += 4

        if is_plot_linear:
            import matplotlib.ticker as ticker
            # ax1.step(np.arange(11), alpha[:11], where='post', lw=3)
            
            ax1.set_ylabel("Отдельный вклад", fontsize=y_size)
            ax1.set_xlabel("t / T", fontsize=x_size)
            ax1.grid()
            ax1.set_xlim(-0.5, 10.5)
            ax1.set_title("Вещественная часть", fontsize=18)

            ax2.set_ylabel("Отдельный вклад", fontsize=y_size)
            ax2.set_xlabel("t / T", fontsize=x_size)
            ax2.grid()
            ax2.set_xlim(-0.5, 10.5)
            ax2.set_title("Мнимая часть", fontsize=18)
            
            ax3.plot((np.arange(80)) / 4 - 2 + 1/2, np.real(linear_signal[:80]), lw=4)
            ax3.set_ylabel("Рез. огибающая", fontsize=y_size)
            ax3.set_xlabel("t / T", fontsize=x_size)
            ax3.grid()
            ax3.set_xlim(-0.5, 10.5)

            ax4.plot((np.arange(80)) / 4 - 2 + 1/2, np.imag(linear_signal[:80]), lw=4)
            ax4.set_ylabel("Рез. огибающая", fontsize=y_size)
            ax4.set_xlabel("t / T", fontsize=x_size)
            ax4.grid()
            ax4.set_xlim(-0.5, 10.5)

            ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax4.xaxis.set_major_locator(ticker.MultipleLocator(1))

            
            plt.tight_layout()
            plt.show()

        # Сдвиг для совмещения: отрезаем 7, чтобы пик был на индексе 3 (конец символьного интервала)
        # (10 - 7 = 3)
        linear_final = linear_signal[6 : num_bits * sps + 6]

        return linear_final
    
    def msk_phase(self, alpha):
        h = self.h
        sps = self.sps
        num_bits = alpha.size
        phi = np.zeros(num_bits)


        for i in range(num_bits):
            alpha_i = alpha[i]
            start_idx = i + 1

            phi[start_idx :] += alpha_i * np.pi * h

        return phi
    
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

            # Линейная
            signal_linear_envelope = self.liner_mod(burst)

            # MSK
            phi_msk = self.msk_phase(alpha)

            is_plot = False
            
            if is_plot:
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
                    
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,6))
                # fig, (ax1, ax2) = plt.subplots(2, 1)
                y_size = 16
                x_size = 16
                num_tap = 148*4

                # ax1.step(np.arange(25), alpha[:25], where="post", lw=4)
                ax1.plot(np.arange(num_tap) / 4,
                      np.real(signal_envelope[:num_tap]), lw=4, label="Нелинейная м.")
                ax1.plot(np.arange(num_tap) / 4,
                      np.real(signal_linear_envelope[:num_tap]), lw=4, label="Линеаризованная м.")
                ax1.set_ylabel("Re", fontsize=y_size)
                ax1.set_xlabel("t / T", fontsize=x_size)
                ax1.grid()
                ax1.set_xlim(-0.5, 10.5)
                ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax1.legend(loc="upper right", fontsize=12)

                ax2.plot(np.arange(num_tap) / 4,
                      np.imag(signal_envelope[:num_tap]), lw=4, label="Нелинейная м.")
                ax2.plot(np.arange(num_tap) / 4,
                      np.imag(signal_linear_envelope[:num_tap]), lw=4, label="Линеаризованная м.")
                ax2.set_ylabel("Im", fontsize=y_size)
                ax2.set_xlabel("t / T", fontsize=x_size)
                ax2.grid()
                ax2.set_xlim(-0.5, 10.5)
                ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax2.legend(loc="upper right", fontsize=12)


                # ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=10, borderaxespad=0)
                # ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=10, borderaxespad=0)


                # ax2.plot(np.arange(100) / 4, phi[:100], lw=4)
                # ax2.set_ylabel("Фаза, рад", fontsize=y_size)
                # ax2.set_xlabel("t / T", fontsize=x_size)
                # ax2.grid()
                # ax2.set_xlim(-0.5, 10.5)
                # ax2.set_ylim(-0, 7) 
                # ax2.yaxis.set_major_locator(ticker.MultipleLocator(np.pi / 2))
                # ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
                # ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))


                # ax3.plot(np.arange(25), phi_msk[:25], lw=4)
                
                # ax3.set_ylabel("Фаза MSK, радианы", fontsize=y_size)
                # ax3.set_xlabel("t / T", fontsize=x_size)
                # ax3.grid()
                # ax3.set_xlim(-0.5, 10.5)
                # # ax3.set_ylim(0, 7) 
                # # ax3.yaxis.set_major_locator(ticker.MultipleLocator(np.pi / 2))
                # ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
                # ax3.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))

                plt.tight_layout()
                plt.show()

            is_plot_true_gmsk = False

            if is_plot_true_gmsk:
                shift_plot = 0
                ax_size = 14

                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,6))       
                ax1.step(np.arange(25 - shift_plot), alpha[:25 - shift_plot], where='post', lw=4)
                ax1.set_ylabel("Символы", fontsize=ax_size)
                ax1.set_xlabel("t / T", fontsize=ax_size) 
                ax1.grid()
                ax1.set_xlim(-0.5, 10.5)
                ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))

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

                ax2.plot(np.arange((25 - shift_plot) * 4) / 4, phi[shift_plot * 4 : 100], lw=4)
                ax2.axhline(y=np.pi, color='r', linestyle='--', lw=1)
                ax2.axhline(y=np.pi * 3 / 2, color='r', linestyle='--', lw=1)
                ax2.axhline(y=3*np.pi, color='r', linestyle='--', lw=1)
                ax2.axhline(y=np.pi * 7 / 2, color='r', linestyle='--', lw=1)
                ax2.set_ylabel("Рез. фаза, рад", fontsize=ax_size) 
                ax2.set_xlabel("t / T", fontsize=ax_size) 
                ax2.set_ylim(0, 12)
                ax2.grid()
                ax2.set_xlim(-0.5, 10.5)
                ax2.yaxis.set_major_locator(ticker.MultipleLocator(np.pi/2))
                ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
                ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
                
                ax3.plot(np.arange((25 - shift_plot) * 4) / 4, np.real(signal_envelope[shift_plot * 4 : 100]), lw=4, label="Re")
                ax3.plot(np.arange((25 - shift_plot) * 4) / 4, np.imag(signal_envelope[shift_plot * 4 : 100]), lw=4, label="Im")
                ax3.axhline(y=0, color='r', linestyle='--', lw=1)
                ax3.set_ylabel("Квадратурные сост.", fontsize=ax_size)
                ax3.set_xlabel("t / T", fontsize=ax_size) 
                ax3.grid()
                ax3.set_xlim(-0.5, 10.5)
                ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax3.legend(fontsize=ax_size)

                # plt.suptitle("Со сдвигом") 
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
