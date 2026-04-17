
from config import simulation_params, channel_params
from core.block import Block

from channel.awgn_channel import AWGNChannel
from channel.rayleigh_single_path import RayleighSinglePathChannel
from channel.rayleigh_multipath import RayleighMultipathChannel

class ChannelBlock(Block):
    def __init__(
        self, 
        channel_model: str,
        snr_db: float = None,
        signal_power: float = None, 
        profile: str = None, 
        is_working: bool = True,
    ):
    
        super().__init__(is_working)
    
        self.channel_model = channel_model
        self.sweep_mode = simulation_params.get("sweep_mode", "snr")
       
        self.snr_db = snr_db
        self.signal_power = (
            signal_power 
            if signal_power is not None 
            else channel_params.get("signal_power", None)
        )    
        self.profile = profile if profile is not None else channel_params.get("profile", "TU")

        self.fading_channel = None
        self.noise_channel = AWGNChannel(
            snr_db = self.snr_db,
            temperature = channel_params.get("temperature", 290.0),
            bandwidth = channel_params.get("bandwidth", 200e3),
            noise_figure = channel_params.get("noise_figure", 7.0),
            signal_power = self.signal_power,
            )

        if channel_model == "awgn":
            self.fading_channel = None
            
        elif channel_model == "rayleigh_single":
            self.fading_channel = RayleighSinglePathChannel(
                maximum_doppler_shift = channel_params["doppler"],
                sample_rate = channel_params["sample_rate"]
            )

        elif channel_model == "rayleigh_multipath":
            self.fading_channel = RayleighMultipathChannel(
                sample_rate = channel_params["sample_rate"],
                profile = self.profile,
                maximum_doppler_shift = channel_params["doppler"]
            )
        else:
            raise ValueError(f"Unknown channel model: {channel_model}")

    def set_snr_db(self, value: float):
        self.snr_db = value
        self.noise_channel.snr_db = value
            
    def set_signal_power(self, value: float):
        self.signal_power = value
        self.noise_channel.signal_power = value

    def _process(self, tx_signal):
        if self.fading_channel is not None:
            y = self.fading_channel.process(tx_signal)
        else:
            y = tx_signal

        if self.sweep_mode == "prx":
            return self.noise_channel.process(y, signal_power = self.signal_power)
        else:
            return self.noise_channel.process(y, snr_db = self.snr_db)