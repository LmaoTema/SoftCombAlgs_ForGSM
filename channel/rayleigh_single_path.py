import numpy as np

from channel.types import ChannelState


class DopplerFader:

    def __init__(
        self,
        sample_rate,
        maximum_doppler_shift,
        spectrum = "CLARKE",
        n_sinusoids = 64,
        seed = None,
    ):
        self.fs = float(sample_rate)
        self.fd = float(maximum_doppler_shift)
        self.spectrum = spectrum.upper()
        self.n_sin = int(n_sinusoids)
        self.seed = seed

        if self.spectrum != "IID" and self.n_sin < 8: 
            raise ValueError("n_sinusoids must be >= 8")  # ограничение на минимум синусоид для time-correlated models

        self.rng = np.random.default_rng(seed) # генератор случайных чисел
        self._sample_index = 0 # счётчик времени процесса
        self._init_process()

    def reset(self):
        self._sample_index = 0
        self._init_process()

    @staticmethod
    def _mode_class(spectrum):
        if spectrum in {"IID", "CLARKE"}:
            return "reference"
        return "legacy_non_reference"

    @staticmethod
    def _normalize_average_power(h, target_power = 1.0):
        measured_power = float(np.mean(np.abs(h) ** 2)) if len(h) else 0.0
        if measured_power <= 0:
            return h.astype(np.complex128), measured_power, False
        gain = np.sqrt(float(target_power) / measured_power)
        return (h * gain).astype(np.complex128), measured_power, not np.isclose(gain, 1.0)

    def _init_process(self):
        if self.spectrum == "IID":
            self._freqs = None
            self._coeffs = None
            self._los_phase = None
            self._angles = None
            self._phi = None
            return

        if self.spectrum == "CLARKE":
            self._angles = self.rng.uniform(0.0, 2.0 * np.pi, size = self.n_sin)        # cлучайные углы рассеяния
            self._freqs = self.fd * np.cos(self._angles)        # преобразование углов в допплеровские частоты
            self._coeffs = (
                self.rng.normal(size = self.n_sin) + 1j * self.rng.normal(size = self.n_sin)
            ) / np.sqrt(2.0 * self.n_sin)       # комплексные веса суммы синусоид
            self._los_phase = None
            self._phi = None
            return

        self._freqs = self._draw_frequencies(self.n_sin, self.spectrum)
        self._coeffs = (
            self.rng.normal(size = self.n_sin) + 1j * self.rng.normal(size = self.n_sin)
        ) / np.sqrt(2.0 * self.n_sin)

        self._los_phase = self.rng.uniform(0.0, 2.0 * np.pi)
        self._angles = None
        self._phi = None

    def _draw_frequencies(self, count, spectrum):
        if self.fd <= 0:
            return np.zeros(count, dtype = float)

        if spectrum == "CLASS":
            theta = self.rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size = count)
            return self.fd * np.sin(theta)

        if spectrum == "GAUS1":
            w1 = 1.0
            w2 = 10.0 ** (-10.0 / 10.0)
            p1 = w1 / (w1 + w2)

            mask = self.rng.random(count) < p1
            out = np.empty(count, dtype = float)
            out[mask] = self.rng.normal(-0.8 * self.fd, 0.05 * self.fd, size = np.sum(mask))
            out[~mask] = self.rng.normal(0.4 * self.fd, 0.10 * self.fd, size = np.sum(~mask))
            return np.clip(out, -self.fd, self.fd)

        if spectrum == "GAUS2":
            w1 = 1.0
            w2 = 10.0 ** (-15.0 / 10.0)
            p1 = w1 / (w1 + w2)

            mask = self.rng.random(count) < p1
            out = np.empty(count, dtype = float)
            out[mask] = self.rng.normal(0.7 * self.fd, 0.10 * self.fd, size = np.sum(mask))
            out[~mask] = self.rng.normal(-0.4 * self.fd, 0.15 * self.fd, size = np.sum(~mask))
            return np.clip(out, -self.fd, self.fd)

        if spectrum == "RICE":
            theta = self.rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size = count)
            return self.fd * np.sin(theta)

        raise ValueError(f"Unsupported Doppler spectrum: {spectrum}")

    def generate(self, N):
        h, _ = self.generate_with_metadata(N)
        return h

    def generate_with_metadata(self, N):
        if N <= 0:
            return np.zeros(0, dtype = np.complex128), {
                "fading_mode": self.spectrum,
                "fading_mode_class": self._mode_class(self.spectrum),
                "target_average_channel_power": 1.0,
                "measured_average_channel_power": 0.0,
                "raw_average_channel_power": 0.0,
                "normalization_applied": False,
                "seed": self.seed,
                "fd_hz": self.fd,
                "sample_rate_hz": self.fs,
            }

        '''
        - генерируются `N` независимых комплексных гауссовых отсчётов;
        - деление на `sqrt(2)` делает дисперсии I и Q равными `1/2`;
        - измеренная мощность сохраняется в metadata.
        '''
        if self.spectrum == "IID":
            h = (
                self.rng.normal(size = N) + 1j * self.rng.normal(size = N)
            ) / np.sqrt(2.0)
            self._sample_index += N
            measured_power = float(np.mean(np.abs(h) ** 2))
            return h.astype(np.complex128), {
                "fading_mode": self.spectrum,
                "fading_mode_class": self._mode_class(self.spectrum),
                "target_average_channel_power": 1.0,
                "measured_average_channel_power": measured_power,
                "raw_average_channel_power": measured_power,
                "normalization_applied": False,
                "seed": self.seed,
                "fd_hz": self.fd,
                "sample_rate_hz": self.fs,
            }

        n = np.arange(self._sample_index, self._sample_index + N, dtype = float)

        if self.spectrum == "CLARKE":
            phases = 2.0 * np.pi * np.outer(self._freqs / self.fs, n)           # вычисляются фазы синусоид
            h = np.sum(self._coeffs[:, None] * np.exp(1j * phases), axis = 0)   # cкладывает все синусоиды в один процесс
            h = h - np.mean(h)
            self._sample_index += N
            h_norm, raw_power, normalization_applied = self._normalize_average_power(h) # нормировкаа процесса к средней мощности 1
            return h_norm, {
                "fading_mode": self.spectrum,
                "fading_mode_class": self._mode_class(self.spectrum),
                "target_average_channel_power": 1.0,
                "measured_average_channel_power": float(np.mean(np.abs(h_norm) ** 2)),
                "raw_average_channel_power": raw_power,
                "normalization_applied": normalization_applied,
                "seed": self.seed,
                "fd_hz": self.fd,
                "sample_rate_hz": self.fs,
            }

        phases = 2.0 * np.pi * np.outer(self._freqs / self.fs, n)
        h = np.sum(self._coeffs[:, None] * np.exp(1j * phases), axis = 0)

        if self.spectrum == "RICE":
            k_factor = 1.0
            f_los = 0.7 * self.fd
            los = np.exp(1j * (2.0 * np.pi * f_los * n / self.fs + self._los_phase))
            h = np.sqrt(1.0 / (k_factor + 1.0)) * h + np.sqrt(k_factor / (k_factor + 1.0)) * los

        self._sample_index += N
        h_norm, raw_power, normalization_applied = self._normalize_average_power(h)
        return h_norm, {
            "fading_mode": self.spectrum,
            "fading_mode_class": self._mode_class(self.spectrum),
            "target_average_channel_power": 1.0,
            "measured_average_channel_power": float(np.mean(np.abs(h_norm) ** 2)),
            "raw_average_channel_power": raw_power,
            "normalization_applied": normalization_applied,
            "seed": self.seed,
            "fd_hz": self.fd,
            "sample_rate_hz": self.fs,
        }


class RayleighSinglePathChannel:

    def __init__(
        self,
        maximum_doppler_shift = 0.0,
        sample_rate = 1e6,
        doppler_spectrum = "CLARKE",
        n_sinusoids = 64,
        seed = None,
    ):
        self.fd = float(maximum_doppler_shift)
        self.fs = float(sample_rate)
        self.doppler_spectrum = doppler_spectrum.upper()
        self.n_sinusoids = int(n_sinusoids)
        self.seed = seed

        self._fader = DopplerFader(
            sample_rate = self.fs,
            maximum_doppler_shift = self.fd,
            spectrum = self.doppler_spectrum,
            n_sinusoids = self.n_sinusoids,
            seed = seed,
        )

    def reset(self):
        self._fader.reset()

    def process_with_state(self, x, samples_per_symbol = None):
        x = np.asarray(x, dtype = np.complex128)
        h, fading_metadata = self._fader.generate_with_metadata(len(x))
        y = h * x
        state = ChannelState(
            kind = "flat_fading",
            sample_rate = self.fs,
            samples_per_symbol = samples_per_symbol,
            average_sample_power = float(np.mean(np.abs(y) ** 2)) if len(y) else 0.0,
            average_channel_power = float(np.mean(np.abs(h) ** 2)) if len(h) else 0.0,
            flat_gain = h,
            impulse_response = np.array([1.0 + 0.0j]),
            metadata = {
                "channel_model": "rayleigh_single",
                "doppler_spectrum": self.doppler_spectrum,
                "maximum_doppler_shift": self.fd,
                "fading_mode": fading_metadata["fading_mode"],
                "fading_mode_class": fading_metadata["fading_mode_class"],
                "target_average_channel_power": fading_metadata["target_average_channel_power"],
                "measured_average_channel_power": fading_metadata["measured_average_channel_power"],
                "raw_average_channel_power": fading_metadata["raw_average_channel_power"],
                "normalization_applied": fading_metadata["normalization_applied"],
                "seed": self.seed,
                "fd_hz": self.fd,
                "sample_rate_hz": self.fs,
                "channel_state_power_domain": "physical",
            },
        )
        return y, state

    def process(self, x):
        y, _ = self.process_with_state(x)
        return y
