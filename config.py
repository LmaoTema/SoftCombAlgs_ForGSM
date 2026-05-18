import numpy as np
from transmitter.channel_coder.utils import MSC_PARAMS

simulation_params = {
    "channel_type": "TCHFS",
    "channel_model": "awgn",    # "awgn" / "rayleigh_single" / "rayleigh_multipath"
    "x_axis_metric": "dbm",     # "dbm" / "snr_db" / "ebn0_db"
    "processing_mode": "none",  # "none"/ "half"/ "full"    
    "combining_method": "PDMRC" # "PDMRC" / "SC" / "ACS"
}

block_params = {
    "encoding":       {"is_working": True},
    "interleaver":    {"is_working": True},
    "modulation":     {"is_working": True},
    "channel":        {"is_working": True},
    "matched_filter": {"is_working": True},
    "equalizer":      {"is_working": True},
}

BER = {
    "filename": "BER_to_file",
    
    "h2dB_init": 0,
    "h2dB_init_step": 0.4,
    "h2dB_min_step": 0.1,
    "h2dB_max_step": 1.6,
    "h2dB_max": 100,
    
    "prx_dbm_init": -118.0 + 3 + 0,
    "prx_dbm_init_step": 1.0,
    "prx_dbm_min_step": 1.0,
    "prx_dbm_max_step": 2.0,
    "prx_dbm_max": -110.0,

    "min_BER": 1e-3,
    "min_FER": 1,
    
    "min_NumErBits": 500,
    "min_NumErFrames": 1000,
    "min_NumTrFrames": 1000,
    
    "max_NumTrBits": 5e6,
    "max_NumTrFrames": 2000,
    
    "max_BERRate": 5,
    "min_BERRate": 2,
    "log_language": "Russian",
    
    "stop_by_min_BER": True
}

mode_params = {
    "TCHFS": {
        "scheme": "TCHFS",
        "frame_bits": 260
    },
#    "CS1": {
#        "scheme": "CS1",
#        "frame_bits": 184
#     },
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
    "rect_duration": 1  ,
    "type_demod": "vit_soft" # diff_phase / vit_hard / vit_soft 
}

equalizer_params = {
    "equalizer_type": "MLSE", # ZF / DFE / MLSE
    "channel_model": simulation_params.get("channel_model", "awgn"),
}

# Rs = 1 / T, Fs = sps / T.
DERIVED_SYMBOL_RATE_HZ = 1.0 / modulation_params["T"]
DERIVED_SAMPLE_RATE_HZ = modulation_params["sps"] * DERIVED_SYMBOL_RATE_HZ
PHYSICAL_GSM_CHANNEL_BANDWIDTH_HZ = 200e3

CHANNEL_LINK_BUDGET = {
    "TCHFS": {
        "bits_per_symbol": 1.0,
        "coding_rate": 260.0 / 456.0,
    },
    "CS1": {
        "bits_per_symbol": 1.0,
        "coding_rate": 184.0 / 456.0,
    },
    "MCS1": {
        "bits_per_symbol": 1.0,
        "coding_rate": (MSC_PARAMS["MCS1"]["header_bits"] + MSC_PARAMS["MCS1"]["data_bits"]) / 456.0,
    },
    "MCS5": {
        "bits_per_symbol": 3.0,
        "coding_rate": (MSC_PARAMS["MCS5"]["header_bits"] + MSC_PARAMS["MCS5"]["data_bits"]) / 1384.0,
    },
}

link_budget_params = CHANNEL_LINK_BUDGET.get(
    simulation_params["channel_type"],
    {
        "bits_per_symbol": 1.0,
        "coding_rate": 1.0,
    },
)

channel_params = {
    "temperature": 290.0,       # [К]
    "noise_figure": 7.0,        # [db]
    "physical_channel_bandwidth": PHYSICAL_GSM_CHANNEL_BANDWIDTH_HZ,  # [Hz] 
    "bandwidth": DERIVED_SAMPLE_RATE_HZ,            # [Hz]
    "signal_power": -102.0,                         # принимаемая мощность сигнала [dbm]
    "samples_per_symbol": modulation_params["sps"],
    "bits_per_symbol": link_budget_params["bits_per_symbol"],
    "coding_rate": link_budget_params["coding_rate"],
  
    "profile": "TU",        # TU / RA / HT
    "sample_rate": DERIVED_SAMPLE_RATE_HZ,
    "doppler": 100,
}
