
import numpy as np
from core.base import BasePipeline

class NonePipeline(BasePipeline):

    def prepare_point(self, x_value, ber_ruler=None):

        self.channel.set_signal_power(x_value)
        
    def update_stats(self, ber_ruler, ber_ruler_uncoded, result, tx_bits):

        ber_ruler.update_frame(
            tx_bits=result["tx_bits"],
            rx_bits=result["decoded_bits"],
            channel_output=result.get("channel_output")
        )

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

        rx_output = self.channel.process(tx_signal)

        rx_samples, channel_state, _ = self._unwrap_channel_output(rx_output)

        h = self.estimator.process(rx_samples, tx_signal, channel_state=channel_state)

        mf = self.matched_filter.process(rx_samples, h)

        eq = self.equalizer.process(mf, h)

        llr = self.detector.process(eq, h)

        detected_bits = (llr < 0).astype(np.int8)

        deintl = self.deinterleaver.process(llr)

        decoded_bits = self.decoder.process(deintl)

        return {
            "decoded_bits": decoded_bits,
            "tx_bits": tx_bits,
            "rx_bits": detected_bits,
            "channel_output": rx_output
        }

