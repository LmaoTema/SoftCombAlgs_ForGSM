import numpy as np
from config import block_params


class ProcessingMode:
    NONE = "None"
    HALF = "Half"
    FULL = "Full"


class Pipeline:
    def __init__(self, mode, encoder, interleaver, modulator, channel,
        estimator, matched_filter, equalizer, detector, deinterleaver,
        decoder, soft_llr_generator, combiner=None):
        self.mode = mode

        self.encoder = encoder
        self.interleaver = interleaver
        self.modulator = modulator

        self.channel = channel

        self.estimator = estimator
        self.matched_filter = matched_filter
        self.equalizer = equalizer

        self.detector = detector
        self.deinterleaver = deinterleaver
        self.decoder = decoder

        self.soft_llr_generator = soft_llr_generator
        self.combiner = combiner

    def tx(self, bits: np.ndarray):

        coded = self.encoder.process(bits.tolist())
        coded = self.interleaver.process(coded)

        for i in range (10):
            print(i, coded[i])

        for i in range (142,147):
            print(i, coded[i])

        tx_signal = self.modulator.process(np.array(coded))

        return tx_signal

    def channel_pass(self, tx_signal):
        return self.channel.process(tx_signal)

    def rx(self, rx_signal, tx_signal=None):

        if self.mode == ProcessingMode.HALF:

            llr = self.soft_llr_generator.process(rx_signal)

            if self.combiner is not None:
                llr = self.combiner.process(llr)

            decoded = self.decoder.process(llr)

            return decoded
        
        # Получаем оценку композитного канала
        h = self.estimator.process(rx_signal, tx_signal)
        # Деротируем сигнал и пропускаем через СФ
        mf = self.matched_filter.process(rx_signal, h)

        # Детектор c шумом
        llr = self.detector.process(mf, h)

        # # Детектор без шума
        block_params["channel"]["is_working"] = False
        mf_without = self.matched_filter.process(tx_signal, h)
        lrr_2 = self.detector.process(mf_without, h)

        if self.mode == ProcessingMode.FULL:

            if self.combiner is not None:
                llr = self.combiner.process(llr)

        bits = self.deinterleaver.process(llr)
        bits = self.decoder.process(bits)

        return bits

    def step(self, bits):

        tx_signal = self.tx(bits)
        rx_signal = self.channel_pass(tx_signal)
        rx_out = self.rx(rx_signal, tx_signal)

        return tx_signal, rx_out