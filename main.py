import numpy as np

from core.factory import create_pipeline

from config import (simulation_params, channel_params, mode_params, BER, block_params, modulation_params, equalizer_params)

from transmitter.channel_coder.coder_manager import ChannelCoder
from transmitter.interleaver.inter_manager import Interleaver
from transmitter.modulator import Modulation
from receiver.detector.det_manager import Detector

from channel.channel_manager import ChannelBlock

# from receiver.estimator import ChannelEstimate
from receiver.new_estimator import ChannelEstimate
from receiver.matched_filter import MatchedFilter
from receiver.equalizer.equalizer_manager import Equalizer
from receiver.decoder.dec_manager import ChannelDecoder
from receiver.deinterleaver.deinter_manager import Deinterleaver
from receiver.softcomb.softgen import SoftGenerator
from receiver.softcomb.comb_manager import CombManager

from drawber.berruler import BERRuler
from drawber.berruler_half import HalfBERRuler
from drawber.res_saver import save_ber_results


def build_pipeline(mode, channel_type, mode_cfg):
    combining_method = simulation_params["combining_method"]
    
    encoder = ChannelCoder(channel_type, is_working=block_params["encoding"]["is_working"])
    interleaver = Interleaver(channel_type, is_working=block_params["interleaver"]["is_working"])

    deinterleaver = Deinterleaver(channel_type, is_working=block_params["interleaver"]["is_working"])
    decoder = ChannelDecoder(scheme=mode_cfg["scheme"], vit_mode=modulation_params["type_demod"], combining_method=combining_method, is_working=block_params["encoding"]["is_working"])

    modulator = Modulation(channel_type, modulation_params, is_working=block_params["modulation"]["is_working"])
    detector = Detector(channel_type, modulation_params, block_params, is_working=block_params["modulation"]["is_working"])

    estimator = ChannelEstimate(modulation_params, simulation_params)
    matched_filter = MatchedFilter(modulation_params, is_working=block_params["matched_filter"]["is_working"])

    equalizer = Equalizer(equalizer_params, modulation_params, is_working=block_params["equalizer"]["is_working"])

    soft_llr_generator = SoftGenerator(simulation_params["channel_type"], simulation_params["channel_model"], profile=channel_params.get("profile", "TU"), is_working=True)

    

    if combining_method == "ACS":
        combiner = None
    else:
        combiner = CombManager(method=combining_method)
    
    channel = ChannelBlock(
        channel_model = simulation_params["channel_model"], 
        profile = channel_params.get("profile", "TU"),
        is_working = block_params["channel"]["is_working"],
    )

    return create_pipeline(mode=mode,encoder=encoder, interleaver=interleaver,modulator=modulator,
        channel = channel, estimator=estimator, matched_filter=matched_filter, equalizer=equalizer,
        detector=detector, deinterleaver=deinterleaver, decoder=decoder, soft_llr_generator=soft_llr_generator, combiner=combiner)

def main():

    channel_type = simulation_params["channel_type"]

    axis_metric = simulation_params.get("x_axis_metric", simulation_params.get("sweep_mode", "dbm"))

    processing_mode = simulation_params["processing_mode"]

    mode_cfg = mode_params[channel_type]
    frame_bits = mode_cfg["frame_bits"]
        
    pipeline = build_pipeline(processing_mode, channel_type, mode_cfg)

    if processing_mode == "half":
        rssi_points = (pipeline.soft_llr_generator.rssi_db)
        ber_ruler = HalfBERRuler(rssi_points=rssi_points, channel_type=channel_type, **BER)
        ber_ruler_uncoded = HalfBERRuler(rssi_points=rssi_points, enable_log=False, channel_type=channel_type,**BER)
    else:
        ber_ruler = BERRuler(**BER, channel_type=channel_type, axis_metric=axis_metric)
        ber_ruler_uncoded = BERRuler(**BER, channel_type=channel_type, axis_metric=axis_metric, enable_log=False)
    
    while not ber_ruler.isStop:

        x_value = ber_ruler.current_x

        pipeline.prepare_point(x_value, ber_ruler)

        while not ber_ruler.is_point_finished():

            bits = np.random.randint(0, 2, frame_bits)

            result = pipeline.process(bits)

            pipeline.update_stats(ber_ruler, ber_ruler_uncoded, result, bits)

        ber_ruler.finalize_point()
        if ber_ruler_uncoded is not None:
            ber_ruler_uncoded.finalize_point()

    res_coded = ber_ruler.get_results()
    res_uncoded = ber_ruler_uncoded.get_results()
    
    # передаём результирующие параметры для сохранения
    save_ber_results(res_coded, res_uncoded,simulation_params)
    # plot_ber(res_coded["x"], res_coded["results"], uncoded_results=res_uncoded["results"], channel_type=channel_type, axis_metric=axis_metric,)

    return (
        res_coded["x"],
        res_coded["results"],
        res_uncoded["results"]
    )

if __name__ == "__main__":
    main()
    