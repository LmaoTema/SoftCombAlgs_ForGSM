from transmitter.channel_coder.utils import MSC_PARAMS

simulation_params = {
    "channel_type": "TCHFS",
    "channel_model": "awgn",    # "awgn" / "rayleigh_single" / "rayleigh_multipath"
    "sweep_mode": "snr"         # "prx"/"snr"
}

block_params = {

    "encoding":       {"is_working": True},
    "interleaver":    {"is_working": True},
    "modulation":     {"is_working": True},
    "channel":        {"is_working": True},
    "matched filter": {"is_working": False},
    "equalizer":      {"is_working": False},
}

BER = {
    "h2dB_init": 0.0,
    "h2dB_init_step": 0.4,
    "h2dB_min_step": 0.1,
    "h2dB_max_step": 1.6,
    "h2dB_max": 26.0,
    
    "prx_dbm_init": -118.0,
    "prx_dbm_init_step": 2.0,
    "prx_dbm_min_step": 2.0,
    "prx_dbm_max_step": 4.0,
    "prx_dbm_max": -100.0,

    "min_BER": 1e-3,
    "min_FER": 1,
    
    "min_NumErBits": 7800,
    "min_NumErFrames": 400,
    "min_NumTrFrames": 800,
    
    "max_NumTrBits": 1e8,
    "max_NumTrFrames": 5000,
    
    "max_BERRate": 5,
    "min_BERRate": 2,
    "log_language": "Russian",
    
    "stop_by_min_BER": False
}

mode_params = {
    "TCHFS": {
        "scheme": "TCHFS",
        "frame_bits": 260
    },
   # "CS1": {
    #    "scheme": "CS1",
    #    "frame_bits": 184
    #},
    #"MCS1": {
    #    "scheme": "MCS1",
     #   "frame_bits": MSC_PARAMS["MCS1"]["header_bits"] + MSC_PARAMS["MCS1"]["data_bits"]
    #},
    #"MCS5": {
     #   "scheme": "MCS5",
     #   "frame_bits": MSC_PARAMS["MCS5"]["header_bits"] + MSC_PARAMS["MCS5"]["data_bits"]
    #}
}

modulation_params = {
    "BT": 0.3,
    "T": 3.69e-6,
    "sps": 4,
    "h": 0.5,
    "gaus_duration": 4,
    "rect_duration": 1,
    "type_demod": "vit_hard" # diff_phase / vit_hard / vit_soft 
}

equalizer_params = {
    "equalizer_type": "MLSE", # ZF /  DFE / MLSE
    "channel_model":simulation_params.get("channel_model", "awgn"),
}

channel_params = {
    "snr_db": 20.0,
    "code_rate": 260 / 456,
    "bits_per_symbol": 1,
    "burst_eff": 456 / 624,
    
    "temperature": 290.0,       # [К]
    "noise_figure": 7.0,        # [db]
    "bandwidth": 200e3,         # [Hz]
    "signal_power": -102.0,     # принимаемая мощность сигнала по умолчанию [dbm]
  
    "profile": "TU",        # TU / RA / HT
    "sample_rate": 1e6,
    "doppler": 100,
}

