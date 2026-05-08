import numpy as np

from config import simulation_params, channel_params
from core.block import Block

from channel.awgn_channel import AWGNChannel
from channel.large_scale import ReceivedPowerScaler
from channel.rayleigh_single_path import RayleighSinglePathChannel
from channel.rayleigh_multipath import RayleighMultipathChannel
from channel.types import ChannelOutput, ChannelState


class ChannelChain:

    def __init__(
        self,
        fading_channel = None,
        noise_channel = None,
        large_scale_channel = None,
        samples_per_symbol = 1.0,
        outage_threshold_db = None,
    ):
        self.fading_channel = fading_channel
        self.noise_channel = noise_channel
        self.large_scale_channel = large_scale_channel
        self.samples_per_symbol = float(samples_per_symbol)
        self.outage_threshold_db = outage_threshold_db

    def reset(self):
        if self.fading_channel is not None and hasattr(self.fading_channel, "reset"):
            self.fading_channel.reset()

    def _default_state(self, signal):
        sample_power = float(np.mean(np.abs(signal) ** 2)) if len(signal) else 0.0
        return ChannelState(
            kind = "awgn_only",
            sample_rate = None if self.noise_channel is None else self.noise_channel.sample_rate,
            samples_per_symbol = self.samples_per_symbol,
            symbol_energy = sample_power * self.samples_per_symbol,
            average_sample_power = sample_power,
            average_channel_power = 1.0,
            impulse_response = np.array([1.0 + 0.0j]),
            metadata = {
                "channel_model": "awgn",
                "fading_mode": "NONE",
                "fading_mode_class": "identity",
                "target_average_channel_power": 1.0,
                "measured_average_channel_power": 1.0,
                "normalization_applied": False,
                "fd_hz": 0.0,
                "sample_rate_hz": None if self.noise_channel is None else self.noise_channel.sample_rate,
                "channel_state_power_domain": "physical",
                "impulse_response_kind": "exact_identity",
                "not_full_ltv_impulse_response": False,
            },
        )

    def _channel_model_name(self):
        if self.fading_channel is None:
            return "awgn"
        if isinstance(self.fading_channel, RayleighSinglePathChannel):
            return "rayleigh_single"
        if isinstance(self.fading_channel, RayleighMultipathChannel):
            return "rayleigh_multipath"
        return "unknown"

    @staticmethod
    def _normalize_signal_to_unit_power(signal):
        signal = np.asarray(signal, dtype = np.complex128)
        measured_power_watt = float(np.mean(np.abs(signal) ** 2)) if len(signal) else 0.0
        if measured_power_watt <= 0:
            return signal, measured_power_watt, 1.0, measured_power_watt

        normalization_gain = 1.0 / np.sqrt(measured_power_watt)
        normalized_signal = signal * normalization_gain
        normalized_power_watt = float(np.mean(np.abs(normalized_signal) ** 2)) if len(normalized_signal) else 0.0
        return normalized_signal, measured_power_watt, normalization_gain, normalized_power_watt

    def process(self, x, *, signal_power_dbm = None, axis_metric = "dbm"):
        x_complex = np.asarray(x, dtype = np.complex128)
        y = x_complex
        pre_fading_input_power_watt = float(np.mean(np.abs(x_complex) ** 2)) if len(x_complex) else 0.0
        channel_state = None
        applied_signal_power_watt = None
        measured_signal_power_watt = None

        if self.fading_channel is not None:
            y, channel_state = self.fading_channel.process_with_state(
                y,
                samples_per_symbol = self.samples_per_symbol,
            )
        else:
            channel_state = self._default_state(y)

        post_fading_power_watt = float(np.mean(np.abs(y) ** 2)) if len(y) else 0.0

        target_signal_power_dbm = signal_power_dbm
        if target_signal_power_dbm is None and self.noise_channel is not None:
            target_signal_power_dbm = self.noise_channel.signal_power
        if target_signal_power_dbm is None:
            raise ValueError("signal_power_dbm must be provided for the power-based channel model.")

        scaling_metadata = None
        if self.large_scale_channel is not None:
            y, applied_signal_power_watt, measured_signal_power_watt, scaling_metadata = self.large_scale_channel.process(
                y,
                target_signal_power_dbm,
            )
        else:
            measured_signal_power_watt = float(np.mean(np.abs(y) ** 2)) if len(y) else 0.0
            applied_signal_power_watt = measured_signal_power_watt
            scaling_metadata = {
                "input_power_watt": measured_signal_power_watt,
                "target_power_watt": applied_signal_power_watt,
                "applied_scale_linear": 1.0,
                "output_power_watt": measured_signal_power_watt,
            }

        applied_signal_power_dbm = float(target_signal_power_dbm)
        measured_signal_power_dbm = self.noise_channel.watt_to_dbm(measured_signal_power_watt)

        if self.noise_channel is None:
            raise ValueError("noise_channel must exist for the power-based AWGN model.")

        y_physical, noise_power_watt = self.noise_channel.process(y)
        post_awgn_power_watt = float(np.mean(np.abs(y_physical) ** 2)) if len(y_physical) else 0.0
        noise_power_dbm = self.noise_channel.watt_to_dbm(noise_power_watt)
        noise_variance_per_sample = self.noise_channel.noise_variance_per_sample()

        carrier_to_noise_db = self.noise_channel.compute_cn_db(applied_signal_power_dbm, noise_power_dbm)
        snr_db = self.noise_channel.compute_snr_db(measured_signal_power_watt, noise_power_watt)
        ebn0_db = self.noise_channel.compute_ebn0_db(snr_db)

        average_channel_power = channel_state.average_channel_power
        outage = False
        if self.outage_threshold_db is not None and average_channel_power is not None and average_channel_power > 0:
            outage = self.noise_channel.linear_to_db(average_channel_power) < float(self.outage_threshold_db)

        y_normalized, measured_output_power_watt, normalization_gain, normalized_signal_power_watt = self._normalize_signal_to_unit_power(y_physical)
        measured_output_power_dbm = self.noise_channel.watt_to_dbm(measured_output_power_watt)
        channel_model_name = self._channel_model_name()
        noise_metadata = self.noise_channel.noise_metadata()

        channel_state.symbol_energy = normalized_signal_power_watt * self.samples_per_symbol
        channel_state.average_sample_power = normalized_signal_power_watt
        channel_state.metadata.update({
            "channel_model": channel_model_name,
            "sample_rate_hz": self.noise_channel.sample_rate,
            "noise_model": noise_metadata["noise_model"],
            "noise_bandwidth_hz": noise_metadata["noise_bandwidth_hz"],
            "noise_bandwidth_mode": noise_metadata["noise_bandwidth_mode"],
            "physical_output_power_watt": measured_output_power_watt,
            "post_fading_power_watt": post_fading_power_watt,
            "post_large_scale_power_watt": measured_signal_power_watt,
            "post_awgn_power_watt": post_awgn_power_watt,
            "final_output_power_watt": normalized_signal_power_watt,
            "normalization_gain": normalization_gain,
            "unit_normalization_applied": True,
            "output_signal_kind": "receiver_normalized",
            "physical_reference_available": True,
            "channel_state_power_domain": "receiver_normalized",
            "physical_symbol_energy": measured_output_power_watt * self.samples_per_symbol,
            "ebn0_interpretation": noise_metadata["ebn0_interpretation"],
        })

        measured_sample_snr_db = snr_db
        resolved_axis_metric = self.noise_channel.normalize_axis_metric(axis_metric)
        axis_value = {
            "dbm": applied_signal_power_dbm,
            "snr_db": snr_db,
            "ebn0_db": ebn0_db,
        }[resolved_axis_metric]

        return ChannelOutput(
            signal = y_normalized,
            channel_state = channel_state,
            noise_power_watt = noise_power_watt,
            noise_power_dbm = noise_power_dbm,
            noise_variance_per_sample = noise_variance_per_sample,
            applied_signal_power_watt = applied_signal_power_watt,
            applied_signal_power_dbm = applied_signal_power_dbm,
            measured_signal_power_watt = measured_signal_power_watt,
            measured_signal_power_dbm = measured_signal_power_dbm,
            measured_output_power_watt = measured_output_power_watt,
            measured_output_power_dbm = measured_output_power_dbm,
            normalized_signal_power_watt = normalized_signal_power_watt,
            normalization_gain = normalization_gain,
            snr_db = snr_db,
            ebn0_db = ebn0_db,
            carrier_to_noise_db = carrier_to_noise_db,
            symbol_rate_hz = self.noise_channel.symbol_rate_hz(),
            bit_rate_hz = self.noise_channel.bit_rate_hz(),
            measured_sample_snr_db = measured_sample_snr_db,
            average_channel_power = average_channel_power,
            outage = outage,
            metadata = {
                "axis_metric": resolved_axis_metric,
                "axis_value": axis_value,
                "channel_model": channel_model_name,
                "fading_mode": channel_state.metadata.get("fading_mode"),
                "fading_mode_class": channel_state.metadata.get("fading_mode_class"),
                "sample_rate_hz": self.noise_channel.sample_rate,
                "noise_model": noise_metadata["noise_model"],
                "noise_bandwidth_hz": noise_metadata["noise_bandwidth_hz"],
                "noise_bandwidth_mode": noise_metadata["noise_bandwidth_mode"],
                "physical_output_power_watt": measured_output_power_watt,
                "post_fading_power_watt": post_fading_power_watt,
                "post_large_scale_power_watt": measured_signal_power_watt,
                "post_awgn_power_watt": post_awgn_power_watt,
                "final_output_power_watt": normalized_signal_power_watt,
                "normalization_gain": normalization_gain,
                "unit_normalization_applied": True,
                "output_signal_kind": "receiver_normalized",
                "physical_reference_available": True,
                "measured_average_channel_power": channel_state.metadata.get("measured_average_channel_power"),
                "ebn0_interpretation": noise_metadata["ebn0_interpretation"],
            },
        )


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
        self.axis_metric = simulation_params.get(
            "x_axis_metric",
            simulation_params.get("sweep_mode", "dbm"),
        )

        self.snr_db = snr_db
        self.signal_power = (
            signal_power
            if signal_power is not None
            else channel_params.get("signal_power", None)
        )
        self.profile = profile if profile is not None else channel_params.get("profile", "TU")
        self.samples_per_symbol = channel_params.get("samples_per_symbol", 1.0)
        self.last_output = None

        self.fading_channel = self._build_fading_channel(channel_model)
        self.noise_channel = self._build_noise_channel()
        self.large_scale_channel = self._build_large_scale_channel()
        self.chain = ChannelChain(
            fading_channel = self.fading_channel,
            noise_channel = self.noise_channel,
            large_scale_channel = self.large_scale_channel,
            samples_per_symbol = self.samples_per_symbol,
            outage_threshold_db = channel_params.get("outage_threshold_db", None),
        )

    def _channel_noise_params(self):
        nested_noise = channel_params.get("noise", {})
        return {
            "signal_power": self.signal_power,
            "samples_per_symbol": nested_noise.get(
                "samples_per_symbol",
                channel_params.get("samples_per_symbol", 1.0),
            ),
            "bits_per_symbol": nested_noise.get(
                "bits_per_symbol",
                channel_params.get("bits_per_symbol", 1.0),
            ),
            "coding_rate": nested_noise.get(
                "coding_rate",
                channel_params.get("coding_rate", 1.0),
            ),
            "sample_rate": nested_noise.get(
                "sample_rate",
                channel_params.get("sample_rate", 1.0),
            ),
            "physical_channel_bandwidth": nested_noise.get(
                "physical_channel_bandwidth",
                channel_params.get("physical_channel_bandwidth", channel_params.get("bandwidth", 200e3)),
            ),
            "noise_bandwidth_mode": nested_noise.get(
                "noise_bandwidth_mode",
                channel_params.get("noise_bandwidth_mode", "sample_rate"),
            ),
            "temperature": nested_noise.get("temperature", channel_params.get("temperature", 290.0)),
            "bandwidth": nested_noise.get("bandwidth", channel_params.get("bandwidth", 200e3)),
            "noise_figure": nested_noise.get("noise_figure", channel_params.get("noise_figure", 7.0)),
        }

    def _channel_fading_params(self):
        nested_fading = channel_params.get("fading", {})
        if "doppler_spectrum" in nested_fading:
            doppler_override = nested_fading["doppler_spectrum"]
        else:
            doppler_override = channel_params.get("doppler_spectrum", None)

        return {
            "sample_rate": nested_fading.get("sample_rate", channel_params.get("sample_rate", 1e6)),
            "maximum_doppler_shift": nested_fading.get("doppler", channel_params.get("doppler", 0.0)),
            "profile": nested_fading.get("profile", self.profile),
            "doppler_spectrum": nested_fading.get("doppler_spectrum", channel_params.get("doppler_spectrum", "CLARKE")),
            "doppler_spectrum_override": doppler_override,
            "n_sinusoids": nested_fading.get("n_sinusoids", 64),
            "seed": nested_fading.get("seed", None),
            "filter_length": nested_fading.get("filter_length", 21),
        }

    def _build_noise_channel(self):
        return AWGNChannel(**self._channel_noise_params())

    def _build_large_scale_channel(self):
        return ReceivedPowerScaler()

    def _build_fading_channel(self, channel_model):
        params = self._channel_fading_params()

        if channel_model == "awgn":
            return None

        if channel_model == "rayleigh_single":
            return RayleighSinglePathChannel(
                maximum_doppler_shift = params["maximum_doppler_shift"],
                sample_rate = params["sample_rate"],
                doppler_spectrum = params["doppler_spectrum"],
                n_sinusoids = params["n_sinusoids"],
                seed = params["seed"],
            )

        if channel_model == "rayleigh_multipath":
            return RayleighMultipathChannel(
                sample_rate = params["sample_rate"],
                profile = params["profile"],
                maximum_doppler_shift = params["maximum_doppler_shift"],
                filter_length = params["filter_length"],
                n_sinusoids = params["n_sinusoids"],
                seed = params["seed"],
                doppler_spectrum_override = params["doppler_spectrum_override"],
            )

        raise ValueError(f"Unknown channel model: {channel_model}")

    def set_snr_db(self, value: float):
        self.snr_db = value

    def set_signal_power(self, value: float):
        self.signal_power = value
        self.noise_channel.signal_power = value

    def _process(self, tx_signal):
        self.last_output = self.chain.process(
            tx_signal,
            signal_power_dbm = self.signal_power,
            axis_metric = self.axis_metric,
        )
        return self.last_output
