import numpy as np

from channel.pdp_profiles import CHANNEL_PROFILES
from channel.rayleigh_single_path import DopplerFader
from channel.types import ChannelState


class RayleighMultipathChannel:

    def __init__(
        self,
        sample_rate,
        profile = "TU",
        maximum_doppler_shift = 100.0,
        filter_length = 21,
        n_sinusoids = 64,
        seed = None,
        doppler_spectrum_override = None,
    ):
        self.fs = float(sample_rate)
        self.fd = float(maximum_doppler_shift)
        self.L = int(filter_length)
        self.n_sinusoids = int(n_sinusoids)
        self.seed = seed
        self.tail_policy = "same_length_with_tail_accounting"
        self.doppler_spectrum_override = None if doppler_spectrum_override is None else str(doppler_spectrum_override).upper()

        if self.L % 2 == 0:
            raise ValueError("filter_length must be odd")

        if profile not in CHANNEL_PROFILES:
            raise ValueError(
                f"Unknown profile '{profile}'. Available: {list(CHANNEL_PROFILES.keys())}"
            )

        pdp = CHANNEL_PROFILES[profile]
        self.profile_name = profile
        self.delays_s = np.asarray(pdp["delays_s"], dtype = float)
        self.powers_db = np.asarray(pdp["powers_db"], dtype = float)
        self.doppler_types = list(pdp["doppler_types"])

        powers_lin = 10.0 ** (self.powers_db / 10.0)
        self.path_powers = powers_lin / np.sum(powers_lin)
        self.num_paths = len(self.delays_s)

        self._prepare_delays(seed)
        self.reset()

    def _prepare_delays(self, seed = None):
        delays_samples = self.delays_s * self.fs
        self.requested_delays_samples = delays_samples
        self.int_delays = np.floor(delays_samples).astype(int)
        self.frac_delays = delays_samples - self.int_delays
        self.max_delay = int(np.max(self.int_delays)) + (self.L - 1)

        base_seed = None if seed is None else int(seed)
        self.effective_doppler_types = (
            [self.doppler_spectrum_override] * self.num_paths
            if self.doppler_spectrum_override is not None
            else list(self.doppler_types)
        )

        self._faders = []
        self._tap_seeds = []
        for i, spectrum in enumerate(self.effective_doppler_types):
            tap_seed = None if base_seed is None else (base_seed + 1000 + i)
            self._tap_seeds.append(tap_seed)
            self._faders.append(
                DopplerFader(
                    sample_rate = self.fs,
                    maximum_doppler_shift = self.fd,
                    spectrum = spectrum,
                    n_sinusoids = self.n_sinusoids,
                    seed = tap_seed,
                )
            )

    def _fractional_delay_kernel(self, frac_delay):
        center = self.L // 2
        n = np.arange(self.L, dtype = float) - center
        kernel = np.sinc(n - frac_delay)
        kernel *= np.blackman(self.L)
        kernel /= np.sqrt(np.sum(np.abs(kernel) ** 2))
        return kernel.astype(np.float64)

    def _apply_fractional_delay(self, x, frac_delay):
        if np.isclose(frac_delay, 0.0):
            return x

        kernel = self._fractional_delay_kernel(frac_delay)
        delayed = np.convolve(x, kernel, mode = "full")
        center = self.L // 2
        return delayed[center:center + len(x)]

    def reset(self):
        for fader in self._faders:
            fader.reset()

    def process_with_state(self, x, samples_per_symbol = None):
        x = np.asarray(x, dtype = np.complex128)
        N = len(x)
        y_full = np.zeros(N + self.max_delay, dtype = np.complex128)
        path_gains = []
        tap_metadata = []

        for k in range(self.num_paths):
            int_delay = int(self.int_delays[k])
            frac_delay = float(self.frac_delays[k])
            path_power = float(self.path_powers[k])

            delayed = self._apply_fractional_delay(x, frac_delay)
            fading, fading_metadata = self._faders[k].generate_with_metadata(len(delayed))
            path = np.sqrt(path_power) * fading * delayed
            path_gain = np.sqrt(path_power) * fading
            path_gains.append(path_gain)
            tap_metadata.append({
                **fading_metadata,
                "tap_index": k,
                "tap_seed": self._tap_seeds[k],
                "normalized_path_power": path_power,
            })

            path_start = int_delay
            path_end = min(path_start + len(path), len(y_full))
            y_full[path_start:path_end] += path[:path_end - path_start]

        y = y_full[:N]
        total_output_energy = float(np.sum(np.abs(y_full) ** 2))
        retained_energy = float(np.sum(np.abs(y) ** 2))
        truncated_tail_energy = max(0.0, total_output_energy - retained_energy)
        truncated_tail_fraction = 0.0 if total_output_energy <= 0.0 else truncated_tail_energy / total_output_energy
        impulse_response = np.zeros(self.max_delay + 1, dtype = np.complex128)
        for delay, power in zip(self.int_delays, self.path_powers):
            impulse_response[int(delay)] += np.sqrt(power)
        path_power_time_average = [float(np.mean(np.abs(gain) ** 2)) for gain in path_gains] if path_gains else []

        state = ChannelState(
            kind = "multipath_fading",
            sample_rate = self.fs,
            samples_per_symbol = samples_per_symbol,
            average_sample_power = float(np.mean(np.abs(y) ** 2)) if len(y) else 0.0,
            average_channel_power = float(np.mean(np.sum(np.abs(np.vstack(path_gains)) ** 2, axis = 0))) if path_gains else 0.0,
            impulse_response = impulse_response,
            path_delays_samples = self.int_delays + self.frac_delays,
            path_powers = self.path_powers,
            path_gains = path_gains,
            metadata = {
                "channel_model": "rayleigh_multipath",
                "profile": self.profile_name,
                "maximum_doppler_shift": self.fd,
                "profile_doppler_types": self.doppler_types,
                "doppler_types": self.effective_doppler_types,
                "doppler_spectrum_override": self.doppler_spectrum_override,
                "channel_state_power_domain": "physical",
                "tail_policy": self.tail_policy,
                "total_output_energy": total_output_energy,
                "retained_energy": retained_energy,
                "truncated_tail_energy": truncated_tail_energy,
                "truncated_tail_fraction": truncated_tail_fraction,
                "requested_delays_s": self.delays_s.tolist(),
                "requested_delays_samples": self.requested_delays_samples.tolist(),
                "effective_integer_delays": self.int_delays.tolist(),
                "effective_fractional_delays": self.frac_delays.tolist(),
                "requested_pdp_db": self.powers_db.tolist(),
                "normalized_path_powers": self.path_powers.tolist(),
                "effective_doppler_types": list(self.effective_doppler_types),
                "fractional_delay_kernel_type": "windowed_sinc_blackman",
                "fractional_delay_filter_length": self.L,
                "fractional_delay_group_delay_samples": self.L // 2,
                "path_power_time_average": path_power_time_average,
                "seed": self.seed,
                "impulse_response_kind": "approximate_static_integer_delay_envelope",
                "not_full_ltv_impulse_response": True,
                "pdp_source_status": "requires_source_verification",
                "pdp_profile_family": "COST207_like",
                "tap_fading_metadata": tap_metadata,
            },
        )
        return y, state

    def process(self, x):
        y, _ = self.process_with_state(x)
        return y
