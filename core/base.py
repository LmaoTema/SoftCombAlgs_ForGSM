from channel.types import ChannelOutput


class BasePipeline:

    def __init__(
        self,
        encoder,
        interleaver,
        modulator,
        channel,
        estimator,
        matched_filter,
        equalizer,
        detector,
        deinterleaver,
        decoder,
        soft_llr_generator,
        combiner
    ):

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

    def process(self, bits):

        return self.process_frame(bits)

    def process_frame(self, bits):

        raise NotImplementedError

    def prepare_point(self, x_value, ber_ruler=None):

        self.x_value = x_value

    def channel_pass(self, tx_signal):

        return self.channel.process(tx_signal)

    def _unwrap_channel_output(self, rx_signal):

        if isinstance(rx_signal, ChannelOutput):

            return (
                rx_signal.signal,
                rx_signal.channel_state,
                rx_signal
            )

        return rx_signal, None, None