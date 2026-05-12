from core.base import BasePipeline
import numpy as np

class FullPipeline(BasePipeline):

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

        coded_bits = self.encoder.process(bits.tolist())

        interleaved_bits = np.array(self.interleaver.process(coded_bits))

        tx_signal = self.modulator.process(interleaved_bits)

        tx_bits = interleaved_bits.reshape(-1, 156)[:, :148].reshape(-1)

        # первый канал
        rx_output_1 = self.channel.process(tx_signal)

        rx_samples_1, channel_state_1, _ = self._unwrap_channel_output(rx_output_1)

        h1 = self.estimator.process(rx_samples_1, tx_signal, channel_state=channel_state_1)

        mf1 = self.matched_filter.process(rx_samples_1, h1)

        eq1 = self.equalizer.process(mf1, h1)

        llr1 = self.detector.process(eq1, h1)

        # второй канал (иммитируем приём для второго сектора)
        rx_output_2 = self.channel.process(tx_signal)
        
        rx_samples_2, channel_state_2, _ = self._unwrap_channel_output(rx_output_2)

        h2 = self.estimator.process(rx_samples_2,tx_signal,channel_state=channel_state_2)

        mf2 = self.matched_filter.process(rx_samples_2, h2)

        eq2 = self.equalizer.process(mf2, h2)

        llr2 = self.detector.process(eq2, h2)

        sector_soft_list = [
            {
                "llr": llr1,
                "hard": (llr1 < 0).astype(np.int8)
            },
            {
                "llr": llr2,
                "hard": (llr2 < 0).astype(np.int8)
            }
        ]

        combined_llr = self.combiner.combine(sector_soft_list)

        deintl = self.deinterleaver.process(combined_llr)

        decoded_bits = self.decoder.process(deintl)

        detected_bits = (combined_llr < 0).astype(np.int8)

        return {
            "decoded_bits": decoded_bits,
            "tx_bits": tx_bits,
            "rx_bits": detected_bits,
            "channel_output": rx_output_1
        }

