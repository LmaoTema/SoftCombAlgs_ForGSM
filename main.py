import numpy as np

from core.pipeline import Pipeline
from config import simulation_params, channel_params, mode_params, BER, block_params, modulation_params, equalizer_params

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
    
def main():

    DEBUG_TRACE = True
    TRACE_FRAME = 0
    frame_counter = 0
    
    channel_type = simulation_params["channel_type"]
    channel_model = simulation_params["channel_model"]
    sweep_mode = simulation_params.get("sweep_mode", "snr")
    profile = channel_params.get("profile", "TU")
    
    mode_cfg = mode_params[channel_type]
    frame_bits = mode_params[channel_type]["frame_bits"]
    
    encoder = ChannelCoder(channel_type, is_working=block_params["encoding"]["is_working"])
    interleaver = Interleaver(channel_type, is_working=block_params["interleaver"]["is_working"])
    
    deinterv = Deinterleaver(channel_type, is_working=block_params["interleaver"]["is_working"])
    decoder = ChannelDecoder(scheme=mode_cfg["scheme"], is_working=block_params["encoding"]["is_working"])
    
    modulator = Modulation(channel_type, modulation_params, is_working=block_params["modulation"]["is_working"])
    detector = Detector(channel_type, modulation_params, block_params, is_working=block_params["modulation"]["is_working"])

    estimator = ChannelEstimate(modulation_params, simulation_params)
    match_filter = MatchedFilter(modulation_params, is_working=block_params["matched filter"]["is_working"])

    equalizer = Equalizer(equalizer_params, modulation_params, is_working=block_params["equalizer"]["is_working"])
    
    ber_ruler = BERRuler(**BER, channel_type = channel_type, sweep_mode = sweep_mode)
    ber_ruler_uncoded = BERRuler(**BER,channel_type=channel_type, enable_log=False) 
    
    while not ber_ruler.isStop:

        if sweep_mode == "prx":
            x_value = ber_ruler.prx_dbm 
            channel = ChannelBlock(channel_model = channel_model, signal_power = x_value, profile = profile, is_working = block_params["channel"]["is_working"])

        else:
            x_value = ber_ruler.h2dB
            channel = ChannelBlock(channel_model = channel_model, snr_db = x_value, profile = profile, is_working = block_params["channel"]["is_working"])
        
        while not ber_ruler.is_point_finished():

            bits = np.random.randint(0, 2, frame_bits)

            bits_cd = encoder.process(bits.tolist())
            
            tx_stream = interleaver.process(bits_cd)
            
            tx_signal = modulator.process(np.array(tx_stream))

            rx_signal = channel.process(tx_signal)

            h = estimator.process(rx_signal, tx_signal)

            match_signal = match_filter.process(rx_signal, h)

            eq_signal = equalizer.process(match_signal, h)

            rx_bits = detector.process(eq_signal, h)
            
            bits_deintr = deinterv.process(rx_bits)

            decoded_bits = decoder.process(bits_deintr)

            if DEBUG_TRACE and frame_counter == TRACE_FRAME:
                if sweep_mode == "prx":
                    print(f"P_rx = {x_value:.2f} dBm")
                else:
                    print(f"h2 = {x_value:.2f} dB")
                print("После источника:", bits)
                # print("После кодера:", tx_stream)
                # print("После модулятора", tx_signal)
                # print("После канала", rx_signal)
                # print("После демодулятора:", rx_bits)
                print("Декодированные:", decoded_bits)

            ber_ruler.update_frame(bits, decoded_bits)
            
            ber_ruler_uncoded.update_frame(np.asarray(bits_cd), np.asarray(bits_deintr))
            frame_counter += 1

        ber_ruler.finalize_point()
        ber_ruler_uncoded.finalize_point()
        
    res_coded = ber_ruler.get_results()
    res_uncoded = ber_ruler_uncoded.get_results()

    x_value = res_coded["x"]
    plot_ber(x_value, res_coded["results"], uncoded_results = res_uncoded["results"], channel_type = channel_type, sweep_mode = sweep_mode)
    
    return x_value, res_coded["results"], res_uncoded["results"]

if __name__ == "__main__":
    main()