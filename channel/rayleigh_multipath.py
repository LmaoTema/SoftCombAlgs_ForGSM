import numpy as np

from channel.pdp_profiles import CHANNEL_PROFILES

class _DopplerFader:
    
    def __init__(
        self,
        sample_rate,
        maximum_doppler_shift,
        spectrum = "CLASS",
        n_sinusoids = 64,
        seed = None,
    ):
        self.fs = float(sample_rate)
        self.fd = float(maximum_doppler_shift)
        self.spectrum = spectrum.upper()
        self.n_sin = int(n_sinusoids)

        if self.n_sin < 8:
            raise ValueError("n_sinusoids must be >= 8")

        self.rng = np.random.default_rng(seed)
        self._sample_index = 0
        self._init_process()

    def reset(self):
        self._sample_index = 0
        self._init_process()

    def _init_process(self):
        self._freqs = self._draw_frequencies(self.n_sin, self.spectrum)

        coeffs = (
            self.rng.normal(size = self.n_sin) + 1j * self.rng.normal(size = self.n_sin)
        ) / np.sqrt(2.0 * self.n_sin)

        coeffs /= np.sqrt(np.sum(np.abs(coeffs) ** 2))
        self._coeffs = coeffs
        self._los_phase = self.rng.uniform(0.0, 2.0 * np.pi)

    def _draw_frequencies(self, count, spectrum):
        if self.fd <= 0:
            return np.zeros(count, dtype = float)

        if spectrum == "CLASS":
            # Clarke/Jakes PSD
            theta = self.rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size = count)
            return self.fd * np.sin(theta)

        if spectrum == "GAUS1":
            w1 = 1.0
            w2 = 10.0 ** (-10.0 / 10.0)
            p1 = w1 / (w1 + w2)

            mask = self.rng.random(count) < p1
            out = np.empty(count, dtype = float)
            out[mask] = self.rng.normal(-0.8 * self.fd, 0.05 * self.fd, size=np.sum(mask))
            out[~mask] = self.rng.normal(0.4 * self.fd, 0.10 * self.fd, size=np.sum(~mask))
            return np.clip(out, -self.fd, self.fd)

        if spectrum == "GAUS2":
            w1 = 1.0
            w2 = 10.0 ** (-15.0 / 10.0)
            p1 = w1 / (w1 + w2)

            mask = self.rng.random(count) < p1
            out = np.empty(count, dtype = float)
            out[mask] = self.rng.normal(0.7 * self.fd, 0.10 * self.fd, size=np.sum(mask))
            out[~mask] = self.rng.normal(-0.4 * self.fd, 0.15 * self.fd, size=np.sum(~mask))
            return np.clip(out, -self.fd, self.fd)

        if spectrum == "RICE":
            theta = self.rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size = count)
            return self.fd * np.sin(theta)

        raise ValueError(f"Unsupported Doppler spectrum: {spectrum}")

    def generate(self, N):
        if N <= 0:
            return np.zeros(0, dtype = np.complex128)

        n = np.arange(self._sample_index, self._sample_index + N, dtype=float)
        phases = 2.0 * np.pi * np.outer(self._freqs / self.fs, n)
        h = np.sum(self._coeffs[:, None] * np.exp(1j * phases), axis = 0)

        if self.spectrum == "RICE":
            K = 1.0
            f_los = 0.7 * self.fd
            los = np.exp(1j * (2.0 * np.pi * f_los * n / self.fs + self._los_phase))
            h = np.sqrt(1.0 / (K + 1.0)) * h + np.sqrt(K / (K + 1.0)) * los

        self._sample_index += N
        return h.astype(np.complex128)
    
       
class RayleighMultipathChannel:

    def __init__(
        self, 
        sample_rate, 
        profile = "TU", 
        maximum_doppler_shift = 100.0, 
        filter_length = 21,
        n_sinusoids = 64,
        seed = None,
    ):
        self.fs = float(sample_rate)
        self.fd = float(maximum_doppler_shift)
        self.L = int(filter_length)
        self.n_sinusoids = int(n_sinusoids)
        
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
        self.int_delays = np.floor(delays_samples).astype(int)
        self.frac_delays = delays_samples - self.int_delays
        self.max_delay = int(np.max(self.int_delays)) + (self.L - 1)
        
        base_seed = None if seed is None else int(seed)
        
        self._faders = []
        for i, spectrum in enumerate(self.doppler_types):
            tap_seed = None if base_seed is None else (base_seed + 1000 + i)
            self._faders.append(
             _DopplerFader(
                    sample_rate = self.fs,
                    maximum_doppler_shift = self.fd,
                    spectrum = spectrum,
                    n_sinusoids = self.n_sinusoids,
                    seed = tap_seed,
            )
        )    

    def reset(self):
        for fader in self._faders:
            fader.reset()

    def process(self, x):
        x = np.asarray(x, dtype = np.complex128)
        N = len(x)

        y_full = np.zeros(N + self.max_delay, dtype = np.complex128)

        for k in range(self.num_paths):
            int_delay = int(self.int_delays[k])
            frac_delay = float(self.frac_delays[k])
            path_power = float(self.path_powers[k])

            delayed = self._apply_fractional_delay(x, frac_delay)
            fading = self._faders[k].generate(len(delayed))

            
            path = np.sqrt(path_power) * fading * delayed
            y_full[int_delay:int_delay + len(path)] += path

        return y_full[:N]