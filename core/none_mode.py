
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
        interleaved_bits[4:8] = np.zeros(4)
        tx_signal, tx_linear_signal = self.modulator.process(interleaved_bits)

        is_compare_signal = True

        if is_compare_signal:
            label_size = 12
            y_size = 16
            x_size = 16
            title_size = 20

            left_limit = 0
            right_limit = 1100
            num_tap = right_limit - left_limit

            fig, (ax1, ax2) = plt.subplots(2, 1)

            ax1.plot(np.arange(num_tap) / 4 + left_limit / 4,
                      np.real(tx_signal[left_limit:right_limit]), label="True GMSK", lw=3)
            ax1.plot(np.arange(num_tap) / 4 + left_limit / 4,
                      np.real(tx_linear_signal[left_limit:right_limit]), label="Linearised GMSK", lw=3)
            ax1.grid()
            ax1.set_ylabel("Re", fontsize=y_size)
            ax1.set_xlabel("t / T", fontsize=x_size)
            ax1.legend(fontsize=label_size)

            ax2.plot(np.arange(num_tap) / 4 + left_limit / 4,
                      np.imag(tx_signal[left_limit:right_limit]), label="True GMSK", lw=3)
            ax2.plot(np.arange(num_tap) / 4 + left_limit / 4,
                      np.imag(tx_linear_signal[left_limit:right_limit]), label="Linearised GMSK", lw=3)
            ax2.grid()
            ax2.set_ylabel("Im", fontsize=y_size)
            ax2.set_xlabel("t / T", fontsize=x_size)
            ax2.legend(fontsize=label_size)

            plt.suptitle("Signal envelope", fontsize=title_size)
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
        h = self.estimator.process(rx_signal, tx_signal, channel_state=channel_state)
        mf = self.matched_filter.process(rx_signal, h)
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

