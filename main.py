import numpy as np
import matplotlib.pyplot as plt

from core.factory import create_pipeline

from config import (simulation_params, channel_params, mode_params, BER, block_params, modulation_params, equalizer_params)

from transmitter.channel_coder.coder_manager import ChannelCoder
from transmitter.interleaver.inter_manager import Interleaver
from transmitter.modulator import Modulation
from receiver.detector.det_manager import Detector

from channel.channel_manager import ChannelBlock

from receiver.estimator import ChannelEstimate
from receiver.matched_filter import MatchedFilter
from receiver.equalizer.equalizer_manager import Equalizer
from receiver.decoder.dec_manager import ChannelDecoder
from receiver.deinterleaver.deinter_manager import Deinterleaver
from receiver.softcomb.softgen import SoftGenerator
from receiver.softcomb.comb_manager import CombManager

from drawber.berruler import BERRuler
from drawber.berruler_half import HalfBERRuler
from drawber.plot import plot_ber


def build_pipeline(mode, channel_type, mode_cfg):

    encoder = ChannelCoder(channel_type, is_working=block_params["encoding"]["is_working"])
    interleaver = Interleaver(channel_type, is_working=block_params["interleaver"]["is_working"])

    deinterleaver = Deinterleaver(channel_type, is_working=block_params["interleaver"]["is_working"])
    decoder = ChannelDecoder(scheme=mode_cfg["scheme"], vit_mode=modulation_params["type_demod"], is_working=block_params["encoding"]["is_working"])

    modulator = Modulation(channel_type, modulation_params, is_working=block_params["modulation"]["is_working"])
    detector = Detector(channel_type, modulation_params, block_params, is_working=block_params["modulation"]["is_working"])

    estimator = ChannelEstimate(modulation_params, simulation_params)
    matched_filter = MatchedFilter(modulation_params, is_working=block_params["matched_filter"]["is_working"])

    equalizer = Equalizer(equalizer_params, modulation_params, is_working=block_params["equalizer"]["is_working"])

    soft_llr_generator = SoftGenerator(simulation_params["channel_type"], simulation_params["channel_model"], profile=channel_params.get("profile", "TU"), is_working=True)

    combiner = CombManager(method=simulation_params["combining_method"])

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
        ber_ruler = HalfBERRuler(rssi_points=rssi_points,**BER)
        ber_ruler_uncoded = HalfBERRuler(rssi_points=rssi_points,enable_log=False,**BER)
    else:
        ber_ruler = BERRuler(**BER, channel_type=channel_type, axis_metric=axis_metric)
        ber_ruler_uncoded = BERRuler(**BER, channel_type=channel_type, axis_metric=axis_metric, enable_log=False)

    llr_0 = []
    llr_1 = []
    counter = 0
    
    while not ber_ruler.isStop:

        x_value = ber_ruler.current_x

        pipeline.prepare_point(x_value, ber_ruler)

        while not ber_ruler.is_point_finished():

            bits = np.random.randint(0, 2, frame_bits)

            result = pipeline.process(bits)

            llr_0.append(result["llr_0"])
            llr_1.append(result["llr_1"])
            counter += 1

            if counter == 100:
                rx_output = result["channel_output"]
                llr_0 = np.concatenate(llr_0)
                llr_1 = np.concatenate(llr_1)

                # q25_0, q75_0 = np.percentile(llr_0, [25, 75])
                # bin_width_0 = 2 * (q75_0 - q25_0) * len(llr_0) ** (-1/3)
                # bins_0 = int((llr_0.max() - llr_0.min()) / bin_width_0)

                # q25_1, q75_1 = np.percentile(llr_1, [25, 75])
                # bin_width_1 = 2 * (q75_1 - q25_1) * len(llr_1) ** (-1/3)
                # bins_1 = int((llr_1.max() - llr_1.min()) / bin_width_1)
                

                # Рисуем гистограммы
                plt.hist(llr_0, bins=256, range=(-128, 128), density=True, color='blue', alpha=0.6, label='LLR для 0')
                plt.hist(llr_1, bins=256, range=(-128, 128), density=True, color='red', alpha=0.6, label='LLR для 1')

                # Оформление
                plt.title(f"Распределение LLR на {int(rx_output.applied_signal_power_dbm)} дБм, {int(rx_output.ebn0_db)} дБ")
                plt.xlabel("Значение LLR")
                plt.ylabel("Плотность вероятности")
                plt.grid()
                plt.legend()
                # plt.xlim(-2.5, 2.5)
                plt.show()

            pipeline.update_stats(ber_ruler, ber_ruler_uncoded, result, bits)

        ber_ruler.finalize_point()
        if ber_ruler_uncoded is not None:
            ber_ruler_uncoded.finalize_point()

    res_coded = ber_ruler.get_results()
    res_uncoded = ber_ruler_uncoded.get_results()

    plot_ber(res_coded["x"], res_coded["results"], uncoded_results=res_uncoded["results"], channel_type=channel_type, axis_metric=axis_metric,)

    return (
        res_coded["x"],
        res_coded["results"],
        res_uncoded["results"]
    )

if __name__ == "__main__":
    main()
    