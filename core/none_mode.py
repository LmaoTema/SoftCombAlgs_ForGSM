
import numpy as np
from core.base import BasePipeline

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

        # Канал
        rx_output = self.channel.process(tx_signal)
        ebn0_val = getattr(rx_output, 'ebn0_db', 0)

        # Приемник
        rx_signal, channel_state, _ = self._unwrap_channel_output(rx_output)
        rx_aligned, h = self.estimator.process(rx_signal, tx_signal)
        mf = self.matched_filter.process(rx_aligned, h)

        eq_out = self.equalizer.process(mf, h)

        # Эквалайзер (когда включён) возвращает пару (signal, llr).
        # Когда выключен, Block.process возвращает исходный сигнал mf без llr.
        if isinstance(eq_out, tuple):
            eq_signal, eq_llr = eq_out
        else:
            eq_signal, eq_llr = eq_out, None

        eq_provides_soft = (
            getattr(self.equalizer, "is_working", False)
            and getattr(self.equalizer, "provides_soft", False)
            and eq_llr is not None
        )

        if eq_provides_soft:

            dfe_bits = getattr(self.equalizer.equalizer, "last_hard_bits", None)
            if dfe_bits is not None and len(dfe_bits) > 0:
                detected_bits = dfe_bits
            else:
                detected_bits, _ = self.detector.process(eq_signal, h, ebn0_val)
            llr = eq_llr
        else:
            detected_bits, llr = self.detector.process(eq_signal, h, ebn0_val)

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
            "channel_output": rx_output
        }

