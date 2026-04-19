import numpy as np

from core.pipeline import Pipeline, ProcessingMode

from config import (simulation_params, channel_params, mode_params, BER, block_params, modulation_params, equalizer_params)

from transmitter.channel_coder.coder_manager import ChannelCoder
from transmitter.interleaver.inter_manager import Interleaver
from receiver.decoder.dec_manager import ChannelDecoder
from receiver.deinterleaver.deinter_manager import Deinterleaver

from transmitter.modulator import Modulation
from receiver.detector.det_manager import Detector

from channel.channel_manager import ChannelBlock

from receiver.estimator import ChannelEstimate
from receiver.matched_filter import MatchedFilter
from receiver.equalizer.equalizer_manager import Equalizer

from drawber.berruler import BERRuler
from drawber.plot import plot_ber


def build_pipeline(mode, channel_type, mode_cfg):

    encoder = ChannelCoder(channel_type, is_working=block_params["encoding"]["is_working"])
    interleaver = Interleaver(channel_type, is_working=block_params["interleaver"]["is_working"])

    deinterleaver = Deinterleaver(channel_type, is_working=block_params["interleaver"]["is_working"])
    decoder = ChannelDecoder(scheme=mode_cfg["scheme"], is_working=block_params["encoding"]["is_working"])

    modulator = Modulation(channel_type, modulation_params, is_working=block_params["modulation"]["is_working"])
    detector = Detector(channel_type, modulation_params, block_params, is_working=block_params["modulation"]["is_working"])

    estimator = ChannelEstimate(modulation_params, simulation_params)
    matched_filter = MatchedFilter(modulation_params, is_working=block_params["matched_filter"]["is_working"])

    equalizer = Equalizer(equalizer_params, modulation_params, is_working=block_params["equalizer"]["is_working"])

    soft_llr_generator = Detector(channel_type, modulation_params, block_params, is_working=True)

    combiner = None

    channel = ChannelBlock(channel_model=simulation_params["channel_model"], profile=channel_params.get("profile", "TU"),is_working=block_params["channel"]["is_working"])

    return Pipeline(mode=mode,encoder=encoder, interleaver=interleaver,modulator=modulator,
        channel=channel, estimator=estimator, matched_filter=matched_filter, equalizer=equalizer,
        detector=detector, deinterleaver=deinterleaver, decoder=decoder, soft_llr_generator=soft_llr_generator, combiner=combiner)

def main():

    channel_type = simulation_params["channel_type"]
    sweep_mode = simulation_params.get("sweep_mode", "snr")
    mode_cfg = mode_params[channel_type]
    frame_bits = mode_cfg["frame_bits"]

    mode = ProcessingMode.FULL 
    pipeline = build_pipeline(mode, channel_type, mode_cfg)

    ber_ruler = BERRuler(**BER, channel_type=channel_type, sweep_mode=sweep_mode)
    ber_ruler_uncoded = BERRuler(**BER, channel_type=channel_type, enable_log=False)

    while not ber_ruler.isStop:
        if sweep_mode == "prx":
            x_value = ber_ruler.prx_dbm
            pipeline.channel.signal_power = x_value
        else:
            x_value = ber_ruler.h2dB
            pipeline.channel.snr_db = x_value

        while not ber_ruler.is_point_finished():

            bits = np.random.randint(0, 2, frame_bits)
            tx_signal = pipeline.tx(bits)
            rx_signal = pipeline.channel_pass(tx_signal)
            decoded_bits = pipeline.rx(rx_signal, tx_signal)
            ber_ruler.update_frame(bits, decoded_bits)

        ber_ruler.finalize_point()
        ber_ruler_uncoded.finalize_point()

    res_coded = ber_ruler.get_results()
    res_uncoded = ber_ruler_uncoded.get_results()

    x_value = res_coded["x"]

    plot_ber(x_value, res_coded["results"], uncoded_results=res_uncoded["results"], channel_type=channel_type,sweep_mode=sweep_mode )

    return x_value, res_coded["results"], res_uncoded["results"]

if __name__ == "__main__":
    main()