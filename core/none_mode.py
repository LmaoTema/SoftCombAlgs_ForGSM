
import numpy as np
from core.base import BasePipeline
import matplotlib.pyplot as plt

class NonePipeline(BasePipeline):

    def prepare_point(self, x_value, ber_ruler=None):

        self.channel.set_signal_power(x_value)
        
    def update_stats(self, ber_ruler, ber_ruler_uncoded, result, bits):
        # Coded BER
        ber_ruler.update_frame(
            bits,
            rx_bits=result["decoded_bits"],
            channel_output=result.get("channel_output")
        )
        
        # Uncoded BER
        ber_ruler_uncoded.update_frame(
            tx_bits=result["tx_bits"],
            rx_bits=result["rx_bits"],
            channel_output=result.get("channel_output")
        )

    def process_frame(self, bits):

        # Передатчик
        coded_bits = self.encoder.process(bits.tolist())
        interleaved_bits = np.array(self.interleaver.process(coded_bits))
        
        # Для комбинации 0001000
        # interleaved_bits[4:8] = np.zeros(4)
        tx_signal, tx_linear_signal = self.modulator.process(interleaved_bits)

        is_compare_signal = False

        if is_compare_signal:

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

            label_size = 12 
            y_size = 16
            x_size = 14
            title_size = 20

            num_tap = len(tx_signal)

            ax1.plot(np.arange(num_tap) / 4,
                      np.real(tx_signal[:num_tap]), label="GMSK", lw=3)
            ax1.plot(np.arange(num_tap) / 4,
                      np.real(tx_linear_signal[:num_tap]), label="Линеаризованная GMSK", lw=3)
            ax1.grid()
            ax1.set_ylabel("Вещественная часть", fontsize=y_size)
            ax1.set_xlabel("t / T", fontsize=x_size)
            ax1.legend(fontsize=label_size, loc = "upper right")
            ax1.set_xlim(-0.5, 148.5)

            ax2.plot(np.arange(num_tap) / 4,
                      np.imag(tx_signal[:num_tap]), label="GMSK", lw=3)
            ax2.plot(np.arange(num_tap) / 4,
                      np.imag(tx_linear_signal[:num_tap]), label="Линеаризованная GMSK", lw=3)
            ax2.grid()
            ax2.set_ylabel("Мнимая часть", fontsize=y_size)
            ax2.set_xlabel("t / T", fontsize=x_size)
            ax2.legend(fontsize=label_size, loc = "upper right")
            ax2.set_xlim(-0.5, 148.5)

            plt.tight_layout()
            plt.show()


        tx_bits = interleaved_bits.reshape(-1, 156)[:, :148].reshape(-1)

        is_print = False
        
        if is_print:
            for i in range (8):
                print('______________________')
                print('number = ', i, 'bit = ', tx_bits[i])

            for i in range (144, 148):
                print('______________________')
                print('number = ', i, 'bit = ', tx_bits[i])

        # Канал
        rx_output = self.channel.process(tx_signal)
        ebn0_val = getattr(rx_output, 'ebn0_db', 0)

        # Приемник
        rx_signal, channel_state, _ = self._unwrap_channel_output(rx_output)

        is_plot_rx = False

        if is_plot_rx:
            fig, (ax1, ax2) = plt.subplots(2,1)

            ax1.plot(np.arange(40) / 4, rx_signal[:40], lw=3)
            ax1.set_ylabel("rx signal")
            ax1.set_xlabel("t / T")
            ax1.grid()

            ax2.plot(np.arange(len(rx_signal)) / 4, rx_signal, lw=3)
            ax2.set_ylabel("all rx signal")
            ax2.set_xlabel("t / T")
            ax2.grid()

            plt.tight_layout()
            plt.show()


        h = self.estimator.process(rx_signal, tx_signal, channel_state=channel_state)
        mf = self.matched_filter.process(rx_signal, h)
        
        is_plot_mf = False

        if is_plot_mf:
            import matplotlib.ticker as ticker # Добавляем импорт для управления шагом

            fig, (ax1, ax2, ax3) = plt.subplots(3,1)

            axis_size = 14
            label_size = 8
            num_sampels = 20

            # Устанавливаем шаг 1 для всех осей
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))

            ax1.plot(np.arange(num_sampels) , np.real(rx_signal[:num_sampels]), lw=3, label = "before MF")
            ax1.plot(np.arange(num_sampels) , np.real(mf[:num_sampels]), lw=3, label = "after MF")
            ax1.set_ylabel("Re", fontsize=axis_size) 
            ax1.set_xlabel("t / T", fontsize=axis_size) 
            ax1.grid()
            ax1.legend(fontsize=label_size, loc = "upper right")

            ax2.plot(np.arange(num_sampels) , np.imag(rx_signal[:num_sampels]), lw=3, label = "before MF")
            ax2.plot(np.arange(num_sampels) , np.imag(mf[:num_sampels]), lw=3, label = "after MF")
            ax2.set_ylabel("Im", fontsize=axis_size) 
            ax2.set_xlabel("t / T", fontsize=axis_size) 
            ax2.grid()
            ax2.legend(fontsize=label_size, loc = "upper right")

            ax3.plot(np.arange(num_sampels) , np.abs(rx_signal[:num_sampels]), lw=3, label = "before MF")
            ax3.plot(np.arange(num_sampels) , np.abs(mf[:num_sampels]), lw=3, label = "after MF")
            ax3.set_ylabel("Abs", fontsize=axis_size) 
            ax3.set_xlabel("t / T", fontsize=axis_size) 
            ax3.grid()
            ax3.legend(fontsize=label_size, loc = "upper right")
            ax3.set_ylim(0, 3)

            print(mf[3], mf[7])
            print(mf[2], mf[6])
            print(mf[4], mf[8])

            plt.tight_layout()
            plt.show()


        eq = self.equalizer.process(mf, h)
        detected_bits, llr, detector_merge_distances = self.detector.process(eq, h, ebn0_val)

        # Графики для проверки llrов
        llr_0 = llr[tx_bits == 0]
        llr_1 = llr[tx_bits == 1]

        # Если есть мягкие решения, то в перемежитель подаем llr
        if self.detector.detector.type_demod == "vit_soft":
            deintl = self.deinterleaver.process(llr)
        else:
            deintl = self.deinterleaver.process(detected_bits)

        decoded_bits = self.decoder.process(deintl)

        return {
            "decoded_bits": decoded_bits,
            "tx_bits": tx_bits,
            "rx_bits": detected_bits,
            "channel_output": rx_output,
            "llr_0": llr_0,
            "llr_1": llr_1,
            "detector_merge_distances": detector_merge_distances
        }

