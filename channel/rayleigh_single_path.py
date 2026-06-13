import numpy as np

from channel.types import ChannelState


class DopplerFader:
    def __init__(
        self,
        sample_rate,
        maximum_doppler_shift,
        spectrum="CLARKE",
        n_sinusoids=64,
        clarke_backend="JAKES",
        seed=None,
    ):
        self.fs = float(sample_rate)
        self.fd = float(maximum_doppler_shift)
        self.spectrum = spectrum.upper()
        self.n_sin = int(n_sinusoids)
        self.clarke_backend = str(clarke_backend).upper()
        self.seed = seed

        if self.spectrum != "IID" and self.n_sin < 8:
            raise ValueError("n_sinusoids must be >= 8")
        if self.spectrum == "CLARKE" and self.clarke_backend not in {"JAKES", "SOS", "IDFT"}:
            raise ValueError("clarke_backend must be 'JAKES', 'SOS', or 'IDFT'")

        self.rng = np.random.default_rng(seed)
        self._sample_index = 0
        self._idft_min_block_size = 262144
        self._idft_xfade_len = 512          
        self._idft_bin_power_cache = {}
        self._clarke_buffer = np.zeros(0, dtype=np.complex128)
        self._static_coeff = None
        self._init_process()

    def reset(self):
        self._sample_index = 0
        self._clarke_buffer = np.zeros(0, dtype=np.complex128)
        self._init_process()

    def advance(self, n):

        if n <= 0:
            return
        n = int(n)
        if self.spectrum == "CLARKE" and self.clarke_backend == "IDFT":
            self._generate_clarke_idft(n)  # потребляем буфер и отбрасываем
        else:
            self._sample_index += n

    @staticmethod
    def _mode_class(spectrum):
        if spectrum in {"IID", "CLARKE"}:
            return "reference"
        return "legacy_non_reference"

    @staticmethod
    def _next_power_of_two(value):
        value = int(max(1, value))
        return 1 << (value - 1).bit_length()

    def _init_process(self):
        if self.spectrum == "IID":
            self._freqs = None
            self._coeffs = None
            self._los_phase = None
            self._angles = None
            return

        if self.spectrum == "CLARKE":
            if self.clarke_backend == "IDFT":
                self._freqs = None
                self._coeffs = None
                self._los_phase = None
                self._angles = None
                self._static_coeff = (
                    self.rng.normal() + 1j * self.rng.normal()
                ) / np.sqrt(2.0)
                return

            if self.clarke_backend == "JAKES":
                self._n_osc = self.n_sin
                self._n_total_waves = 4 * self._n_osc + 2
                k = np.arange(1, self._n_osc + 1, dtype=float)
                self._angles = 2.0 * np.pi * k / self._n_total_waves
                self._freqs = self.fd * np.cos(self._angles)
                self._beta = np.pi * k / (self._n_osc + 1.0)
                self._oscillator_phases = self.rng.uniform(0.0, 2.0 * np.pi, size=self._n_osc)
                self._max_doppler_phase = self.rng.uniform(0.0, 2.0 * np.pi)
                self._phi_n = self.rng.uniform(0.0, 2.0 * np.pi)
                self._los_phase = None
                return

            self._angles = self.rng.uniform(0.0, 2.0 * np.pi, size=self.n_sin)
            self._freqs = self.fd * np.cos(self._angles)
            self._coeffs = (
                self.rng.normal(size=self.n_sin) + 1j * self.rng.normal(size=self.n_sin)
            ) / np.sqrt(2.0 * self.n_sin)
            self._los_phase = None
            return

        self._freqs = self._draw_frequencies(self.n_sin, self.spectrum)
        self._coeffs = (
            self.rng.normal(size=self.n_sin) + 1j * self.rng.normal(size=self.n_sin)
        ) / np.sqrt(2.0 * self.n_sin)
        self._los_phase = self.rng.uniform(0.0, 2.0 * np.pi)
        self._angles = None

    def _draw_frequencies(self, count, spectrum):
        if self.fd <= 0:
            return np.zeros(count, dtype=float)

        if spectrum == "CLASS":
            theta = self.rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=count)
            return self.fd * np.sin(theta)

        if spectrum == "GAUS1":
            w1 = 1.0
            w2 = 10.0 ** (-10.0 / 10.0)
            p1 = w1 / (w1 + w2)

            mask = self.rng.random(count) < p1
            out = np.empty(count, dtype=float)
            out[mask] = self.rng.normal(-0.8 * self.fd, 0.05 * self.fd, size=np.sum(mask))
            out[~mask] = self.rng.normal(0.4 * self.fd, 0.10 * self.fd, size=np.sum(~mask))
            return np.clip(out, -self.fd, self.fd)

        if spectrum == "GAUS2":
            w1 = 1.0
            w2 = 10.0 ** (-15.0 / 10.0)
            p1 = w1 / (w1 + w2)

            mask = self.rng.random(count) < p1
            out = np.empty(count, dtype=float)
            out[mask] = self.rng.normal(0.7 * self.fd, 0.10 * self.fd, size=np.sum(mask))
            out[~mask] = self.rng.normal(-0.4 * self.fd, 0.15 * self.fd, size=np.sum(~mask))
            return np.clip(out, -self.fd, self.fd)

        if spectrum == "RICE":
            theta = self.rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=count)
            return self.fd * np.sin(theta)

        raise ValueError(f"Unsupported Doppler spectrum: {spectrum}")

    def _clarke_bin_powers(self, fft_size):
        if fft_size in self._idft_bin_power_cache:
            return self._idft_bin_power_cache[fft_size]

        if self.fd <= 0.0:
            powers = np.zeros(fft_size, dtype=float)
            powers[0] = 1.0
            self._idft_bin_power_cache[fft_size] = powers
            return powers

        freq = np.fft.fftfreq(fft_size, d=1.0 / self.fs)
        df = self.fs / fft_size
        left_edges = np.clip((freq - 0.5 * df) / self.fd, -1.0, 1.0)
        right_edges = np.clip((freq + 0.5 * df) / self.fd, -1.0, 1.0)
        bin_powers = (np.arcsin(right_edges) - np.arcsin(left_edges)) / np.pi
        outside = (freq + 0.5 * df <= -self.fd) | (freq - 0.5 * df >= self.fd)
        bin_powers[outside] = 0.0
        bin_powers = np.maximum(bin_powers, 0.0)

        total_power = float(np.sum(bin_powers))
        if total_power <= 0.0:
            raise ValueError("Clarke IDFT spectrum has zero total power.")

        bin_powers /= total_power
        self._idft_bin_power_cache[fft_size] = bin_powers
        return bin_powers

    def _generate_clarke_idft_block(self, min_length):
        fft_size = self._next_power_of_two(max(min_length, self._idft_min_block_size))
        bin_powers = self._clarke_bin_powers(fft_size)
        white_spectrum = (
            self.rng.normal(size=fft_size) + 1j * self.rng.normal(size=fft_size)
        ) / np.sqrt(2.0)
        shaped_spectrum = fft_size * np.sqrt(bin_powers) * white_spectrum
        return np.asarray(np.fft.ifft(shaped_spectrum), dtype=np.complex128)

    def _generate_clarke_idft(self, length):
        if length <= 0:
            return np.zeros(0, dtype=np.complex128)

        if self.fd <= 0.0:
            return np.full(length, self._static_coeff, dtype=np.complex128)

        while len(self._clarke_buffer) < length:
            new_block = self._generate_clarke_idft_block(length - len(self._clarke_buffer))
            if len(self._clarke_buffer) == 0:
                self._clarke_buffer = new_block
            else:
                L = min(self._idft_xfade_len, len(self._clarke_buffer), len(new_block))
                if L > 1:
                    t = np.linspace(0.0, 1.0, L)
                    blended = (
                        np.sqrt(1.0 - t) * self._clarke_buffer[-L:]
                        + np.sqrt(t) * new_block[:L]
                    )
                    self._clarke_buffer = np.concatenate(
                        (self._clarke_buffer[:-L], blended, new_block[L:])
                    )
                else:
                    self._clarke_buffer = np.concatenate((self._clarke_buffer, new_block))

        out = self._clarke_buffer[:length]
        self._clarke_buffer = self._clarke_buffer[length:]
        return out

    def _metadata(self, measured_power):
        return {
            "fading_mode": self.spectrum,
            "fading_mode_class": self._mode_class(self.spectrum),
            "target_average_channel_power": 1.0,
            "measured_average_channel_power": measured_power,
            "raw_average_channel_power": measured_power,
            "normalization_applied": False,
            "seed": self.seed,
            "fd_hz": self.fd,
            "sample_rate_hz": self.fs,
            "clarke_backend": self.clarke_backend if self.spectrum == "CLARKE" else None,
        }

    def generate(self, N):
        h, _ = self.generate_with_metadata(N)
        return h

    def generate_with_metadata(self, N):
        if N <= 0:
            return np.zeros(0, dtype=np.complex128), self._metadata(0.0)

        if self.spectrum == "IID":
            h = (
                self.rng.normal(size=N) + 1j * self.rng.normal(size=N)
            ) / np.sqrt(2.0)
            self._sample_index += N
            measured_power = float(np.mean(np.abs(h) ** 2))
            return h.astype(np.complex128), self._metadata(measured_power)

        if self.spectrum == "CLARKE" and self.clarke_backend == "IDFT":
            h = self._generate_clarke_idft(N)
            self._sample_index += N
            measured_power = float(np.mean(np.abs(h) ** 2))
            return h.astype(np.complex128), self._metadata(measured_power)

        n = np.arange(self._sample_index, self._sample_index + N, dtype=float)

        if self.spectrum == "CLARKE" and self.clarke_backend == "JAKES":
            t = n / self.fs
            phases = 2.0 * np.pi * np.outer(self._freqs, t) + self._oscillator_phases[:, None]
            oscillators = np.cos(phases)
            max_doppler_oscillator = np.cos(2.0 * np.pi * self.fd * t + self._max_doppler_phase)
            in_phase = (
                np.sqrt(2.0) * np.cos(self._phi_n) * max_doppler_oscillator
                + 2.0 * np.sum(np.cos(self._beta)[:, None] * oscillators, axis=0)
            )
            quadrature = (
                np.sqrt(2.0) * np.sin(self._phi_n) * max_doppler_oscillator
                + 2.0 * np.sum(np.sin(self._beta)[:, None] * oscillators, axis=0)
            )
            h = (in_phase + 1j * quadrature) / np.sqrt(2.0 * self._n_osc + 1.0)
            self._sample_index += N
            measured_power = float(np.mean(np.abs(h) ** 2))
            return h.astype(np.complex128), self._metadata(measured_power)

        phases = 2.0 * np.pi * np.outer(self._freqs / self.fs, n)
        h = np.sum(self._coeffs[:, None] * np.exp(1j * phases), axis=0)

        if self.spectrum == "RICE":
            k_factor = 1.0
            f_los = 0.7 * self.fd
            los = np.exp(1j * (2.0 * np.pi * f_los * n / self.fs + self._los_phase))
            h = np.sqrt(1.0 / (k_factor + 1.0)) * h + np.sqrt(k_factor / (k_factor + 1.0)) * los

        self._sample_index += N
        measured_power = float(np.mean(np.abs(h) ** 2))
        return h.astype(np.complex128), self._metadata(measured_power)


class RayleighSinglePathChannel:
    def __init__(
        self,
        maximum_doppler_shift=0.0,
        sample_rate=1e6,
        doppler_spectrum="CLARKE",
        n_sinusoids=64,
        clarke_backend="JAKES",
        seed=None,
    ):
        self.fd = float(maximum_doppler_shift)
        self.fs = float(sample_rate)
        self.doppler_spectrum = doppler_spectrum.upper()
        self.n_sinusoids = int(n_sinusoids)
        self.clarke_backend = str(clarke_backend).upper()
        self.seed = seed

        self._fader = DopplerFader(
            sample_rate=self.fs,
            maximum_doppler_shift=self.fd,
            spectrum=self.doppler_spectrum,
            n_sinusoids=self.n_sinusoids,
            clarke_backend=self.clarke_backend,
            seed=seed,
        )

    def reset(self):
        self._fader.reset()

    def process_with_state(self, x, samples_per_symbol=None):
        x = np.asarray(x, dtype=np.complex128)
        h, fading_metadata = self._fader.generate_with_metadata(len(x))
        y = h * x
        state = ChannelState(
            kind="flat_fading",
            sample_rate=self.fs,
            samples_per_symbol=samples_per_symbol,
            average_sample_power=float(np.mean(np.abs(y) ** 2)) if len(y) else 0.0,
            average_channel_power=float(np.mean(np.abs(h) ** 2)) if len(h) else 0.0,
            flat_gain=h,
            impulse_response=np.array([1.0 + 0.0j]),
            metadata={
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
                "clarke_backend": fading_metadata.get("clarke_backend"),
                "channel_state_power_domain": "physical",
            },
        )
        return y, state

    def process(self, x):
        y, _ = self.process_with_state(x)
        return y
