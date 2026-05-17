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

    def rx_detector(self, rx_output, tx_signal):

        rx_signal, channel_state, _ = self._unwrap_channel_output(rx_output)
        h = self.estimator.process(rx_signal, tx_signal, channel_state=channel_state)
        mf = self.matched_filter.process(rx_signal, h)
        eq = self.equalizer.process(mf, h)
        detected_bits, llr = self.detector.process(eq, h, rx_output.ebn0_db)

        return detected_bits, llr
    
    def process_frame(self, bits):

        # Передатчик
        coded_bits = self.encoder.process(bits.tolist())
        interleaved_bits = np.array(self.interleaver.process(coded_bits))
        tx_signal = self.modulator.process(interleaved_bits)
        tx_bits = interleaved_bits.reshape(-1, 156)[:, :148].reshape(-1)

        # Первый канал
        rx_output_1 = self.channel.process(tx_signal)
        detected_bits_1, llr_1 = self.rx_detector(rx_output_1, tx_signal)

        # Второй канал (иммитируем приём для второго сектора)
        rx_output_2 = self.channel.process(tx_signal)
        detected_bits_2, llr_2 = self.rx_detector(rx_output_2, tx_signal)

        sector_soft_list = [llr_1 , llr_2]
        # Делаем деперемежение для каждого набора данных отдельно 
        sector_deinterleaved = []
        for llr in sector_soft_list:
            deintl_llr = self.deinterleaver.process(llr)     
            sector_deinterleaved.append(deintl_llr)
            
        decoded_bits = self.decoder.process(sector_deinterleaved)

        detected_bits = detected_bits_1

        return {
            "decoded_bits": decoded_bits,
            "tx_bits": tx_bits,
            "rx_bits": detected_bits,
            "channel_output": rx_output_1
        }

