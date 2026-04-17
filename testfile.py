import numpy as np
import matplotlib.pyplot as plt

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

plot_params = {
    "modulation":     {"is_working": False},
    "estimation":     {"is_working": False},
    "matched filter": {"is_working": True},
}
def plot_num_subplot(plots_data):

    num_plot = len(plots_data)
    fig = plt.figure()

    for i in range(num_plot):
        data = plots_data[i]
        ax = fig.add_subplot(num_plot, 1, (i + 1))

        ax.plot(data["x"], data["y"], c=data.get("color", "r"), lw=data.get("lw", 3))
        ax.set_title(data.get("title", ""), fontsize=20)
        ax.set_xlabel(data.get("xlabel", ""), fontsize=14)
        ax.set_ylabel(data.get("ylabel", ""), fontsize=14)

        ax.grid()

    plt.tight_layout()
    plt.show()

    return

def plot_many_on_one(plots_data):

    num_plot = len(plots_data)
    plt.figure()

    for i in range(num_plot):
        data = plots_data[i]

        plt.plot(data["x"], data["y"], c=data.get("color", "r"), lw=data.get("lw", 3), label=data.get("label", ""))
        plt.title(data.get("title", ""), fontsize=20)
        plt.xlabel(data.get("xlabel", ""), fontsize=14)
        plt.ylabel(data.get("ylabel", ""), fontsize=14)

        plt.grid()

    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_modulation(symbols_burst, q, phi, tx_signal):
    sps = 4
    plot_data = [
        # q
        {   
            "title": "q(t)",
            "x": np.arange(q.size) / sps,
            "y": q,
            "color": "g",
            "lw": 2
        },
        # symbols_burst
        {   
            "title": "Symbol",
            "x": np.arange(20 * sps) / sps,
            "y": np.repeat(symbols_burst[:20], sps),
            "color": "r",
            "lw": 2
        },
        # phi
        {   
            "title": "phi",
            "x": np.arange(20 * sps) / sps,
            "y": phi[:20 * sps],
            "color": "b",
            "lw": 2
        },
        # tx_signal
        {   
            "title": "tx_signal",
            "x": np.arange(20),
            "y": tx_signal[:20],
            "color": "c",
            "lw": 2
        }
    ]
    plot_num_subplot(plot_data)

def plot_estimation(h):
    sps = 4
    plot_data = [
        # h
        {   
            "title": "h",
            "x": np.arange(h.size) / sps,
            "y": h,
            "color": "r",
            "lw": 2
        }
    ]
    plot_num_subplot(plot_data)

def plot_mf(rx_signal, conv_signal, match_signal):
    sps = 4
    plot_data = [
        # rx_signal
        {   
            "title": "rx_signal",
            "x": np.arange(rx_signal.size) / sps,
            "y": rx_signal,
            "color": "r",
            "lw": 2,
            "label": "rx_signal"
        },
        # conv_signal
        {   
            "title": "conv_signal",
            "x": np.arange(conv_signal.size) / sps,
            "y": conv_signal,
            "color": "g",
            "lw": 2,
            "label": "conv_signal"
        },
        # match_signal
        {   
            "title": "match_signal",
            "x": np.arange(match_signal.size) / sps,
            "y": match_signal,
            "color": "c",
            "lw": 2,
            "label": "match_signal"
        }
    ]
    plot_num_subplot(plot_data)
    plot_many_on_one(plot_data)

    
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
        
        n_errors = 0

        while not ber_ruler.is_point_finished():

            bits = np.random.randint(0, 2, frame_bits)
            bits_cd = encoder.process(bits.tolist()) 
            tx_stream = interleaver.process(bits_cd)
            
            # Modulation
            symbols_burst = modulator.modulator.differential_encoding(tx_stream[:148])
            q = modulator.modulator.generate_q_gmsk()
            phi = modulator.modulator.calc_phase(symbols_burst, q)
            tx_signal = np.exp(1j * phi)
            if plot_params["modulation"]["is_working"]:
               plot_modulation(symbols_burst, q, phi, tx_signal)

            # Channel
            rx_signal = channel.process(tx_signal)

            # Estimation
            h = estimator.h_awgn()
            if plot_params["estimation"]["is_working"]:
                plot_estimation(h)

            # Matched filter
            conv_signal = np.convolve(rx_signal, np.conj(h[::-1]))
            match_signal = conv_signal[int(h.size / 2) + 1: - int(h.size / 2)]
            if plot_params["matched filter"]["is_working"]:
                plot_mf(rx_signal, conv_signal, match_signal)

            # Detector
            rhh = detector.detector.calc_rhh(h)
            increment = detector.detector.calc_increment(rhh)
            sampled_signal = match_signal[detector.detector.sps - 1 :: detector.detector.sps]
            trans_table, old_path_metrics = detector.detector.calc_metric(increment, sampled_signal, start_state=0)
            best_stop_state = detector.detector.find_best_stop_state(old_path_metrics)
            rx_bits = detector.detector.traceback(trans_table, best_stop_state)

            # Deint + decod
            rx_bits = np.append(rx_bits, np.zeros(624-148))
            bits_deintr = deinterv.process(rx_bits)
            decoded_bits = decoder.process(bits_deintr)

            if DEBUG_TRACE and frame_counter == TRACE_FRAME:
                if sweep_mode == "prx":
                    print(f"P_rx = {x_value:.2f} dBm")
                else:
                    print(f"h2 = {x_value:.2f} dB")
                print(bits)
                print(decoded_bits)

            ber_ruler.update_frame(bits, decoded_bits)
            
            ber_ruler_uncoded.update_frame(np.asarray(bits_cd), np.asarray(bits_deintr))
            frame_counter += 1

            n_errors += np.sum(np.array(bits) != np.array(decoded_bits))
            n_frame = 100
            if frame_counter > n_frame:
                print(n_errors/n_frame)


        ber_ruler.finalize_point()
        ber_ruler_uncoded.finalize_point()
        
    res_coded = ber_ruler.get_results()
    res_uncoded = ber_ruler_uncoded.get_results()

    x_value = res_coded["x"]
    plot_ber(x_value, res_coded["results"], uncoded_results = res_uncoded["results"], channel_type = channel_type, sweep_mode = sweep_mode)
    
    return x_value, res_coded["results"], res_uncoded["results"]

if __name__ == "__main__":
    main()