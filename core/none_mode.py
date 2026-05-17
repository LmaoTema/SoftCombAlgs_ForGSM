
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
        tx_signal = self.modulator.process(interleaved_bits)
        tx_bits = interleaved_bits.reshape(-1, 156)[:, :148].reshape(-1)

        for i in range (8):
            print('______________________')
            print('number = ', i, 'bit = ', tx_bits[i])

        for i in range (144, 148):
            print('______________________')
            print('number = ', i, 'bit = ', tx_bits[i])

        # Канал
        rx_output = self.channel.process(tx_signal)

        # Приемник
        rx_signal, channel_state, _ = self._unwrap_channel_output(rx_output)
        h = self.estimator.process(rx_signal, tx_signal, channel_state=channel_state)
        mf = self.matched_filter.process(rx_signal, h)
        eq = self.equalizer.process(mf, h)
        detected_bits, llr = self.detector.process(eq, h, rx_output.ebn0_db)

        # Графики для проверки llrов
        llr_0 = llr[tx_bits == 0]
        llr_1 = llr[tx_bits == 1]

        # # Рисуем гистограммы
        # plt.hist(llr_0, bins=30, density=True, color='blue', alpha=0.6, label='LLR для 0')
        # plt.hist(llr_1, bins=30, density=True, color='red', alpha=0.6, label='LLR для 1')

        # # Оформление "под куратора"
        # plt.title(f"Распределение LLR на {int(rx_output.applied_signal_power_dbm)} дБм, {int(rx_output.ebn0_db)} дБ")
        # plt.xlabel("Значение LLR")
        # plt.ylabel("Плотность вероятности")
        # plt.grid()
        # plt.legend()
        # # plt.xlim(-2.5, 2.5)
        # plt.show()

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
            "llr_1": llr_1
        }

