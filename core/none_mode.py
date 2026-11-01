import numpy as np
from config import simulation_params
from core.base import BasePipeline
from transmitter.interleaver.inter_manager import Interleaver

class NonePipeline(BasePipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.interference_enabled = simulation_params.get(
            "CCI_enabled", False
        )
        self.interference_power_dbm = simulation_params.get(
            "CCI_power_dbm", -88.0
        )

        if self.interference_enabled:
            self.interferer_interleaver = Interleaver(
            scheme=simulation_params["channel_type"],
            is_working=True,
            CCI_enabled=True
            )
        
    def prepare_point(self, x_value, ber_ruler=None):
        self.channel1.set_signal_power(x_value)

    def update_stats(self, ber_ruler, ber_ruler_uncoded, result, bits):

        ber_ruler.update_frame(
            bits,
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

        if self.interference_enabled:
            # формируем мешающий сигнал
            interferer_bits = np.random.randint(0, 2, len(bits))

            interferer_coded = self.encoder.process(interferer_bits.tolist())
            interferer_interleaved = np.array(
                self.interferer_interleaver.process(interferer_coded)
            )

            interferer_signal = self.modulator.process(interferer_interleaved)

            # Суммирование двух сигналов и добавка шума приёмника
            rx_output = self.channel1.process_with_interference(
                useful_signal=tx_signal,
                interferer_signal=interferer_signal,
                interferer_channel=self.channel2,
                interferer_power_dbm=self.interference_power_dbm
            )

        else:
            rx_output = self.channel1.process(tx_signal)

        ebn0_val = getattr(rx_output, "ebn0_db", 0)

        # Приёмник
        rx_signal, channel_state, _ = self._unwrap_channel_output(rx_output)

        rx_aligned, h = self.estimator.process(rx_signal, tx_signal)
        mf = self.matched_filter.process(rx_aligned, h)

        eq_out = self.equalizer.process(mf, h)

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

            dfe_bits = getattr(
                self.equalizer.equalizer, "last_hard_bits", None
            )

            if dfe_bits is not None and len(dfe_bits) > 0:
                detected_bits = dfe_bits
            else:
                detected_bits, _ = self.detector.process(
                    eq_signal, h, ebn0_val
                )

            llr = eq_llr

        else:
            detected_bits, llr = self.detector.process(
                eq_signal, h, ebn0_val
            )

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