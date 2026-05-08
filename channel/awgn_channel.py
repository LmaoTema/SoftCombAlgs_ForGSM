import numpy as np


class _PowerMath():
    BOLTZMANN = 1.380649e-23    # [J/K]

    def __init__(
        self,
        temperature = 290.0,
        bandwidth = 200e3,
        physical_channel_bandwidth = None,
        noise_bandwidth_mode = "sample_rate",
        noise_figure = 7.0,
        sample_rate = 1.0,
        samples_per_symbol = 1.0,
        bits_per_symbol = 1.0,
        coding_rate = 1.0,
    ):

        self.temperature = float(temperature)
        self.configured_bandwidth = float(bandwidth)
        self.physical_channel_bandwidth = float(
            bandwidth if physical_channel_bandwidth is None else physical_channel_bandwidth
        )
        self.noise_bandwidth_mode = str(noise_bandwidth_mode)
        self.noise_figure = float(noise_figure)
        self.sample_rate = float(sample_rate)
        self.samples_per_symbol = float(samples_per_symbol)
        self.bits_per_symbol = float(bits_per_symbol)
        self.coding_rate = float(coding_rate)
        self.bandwidth = self._resolve_noise_bandwidth_hz()

    def _resolve_noise_bandwidth_hz(self):
        mode = self.noise_bandwidth_mode.lower()
        if mode == "sample_rate":
            return self.sample_rate
        if mode in {"receiver_bandwidth", "physical_channel_bandwidth"}:
            return self.physical_channel_bandwidth
        if mode == "configured_bandwidth":
            return self.configured_bandwidth
        raise ValueError(f"Unsupported noise_bandwidth_mode: {self.noise_bandwidth_mode}")

    # Перевод dB в линейное отношение
    @staticmethod
    def db_to_linear(x_db):
        return 10 ** (float(x_db) / 10.0)

    # Перевод линейного отношения в dB
    @staticmethod
    def linear_to_db(x_lin):
        if x_lin <= 0:
            return -np.inf
        return 10.0 * np.log10(float(x_lin))

    # Перевод абсолютной мощности
    @staticmethod
    def dbm_to_watt(p_dbm):
        return 10 ** ((float(p_dbm) - 30.0) / 10.0)

    @staticmethod
    def watt_to_dbm(p_watt):
        if p_watt <= 0:
            return -np.inf
        return 10.0 * np.log10(float(p_watt)) + 30.0

    # Проверка, что число положительно
    @staticmethod
    def _validate_positive(value, name):
        if value is None or float(value) <= 0:
            raise ValueError(f"{name} must be positive.")
        return float(value)

    # Гарантирует, что сигнал всегда рассматривается как `np.complex128`
    @staticmethod
    def as_complex_array(x):
        return np.asarray(x, dtype = np.complex128)
    
    # Перевод noise figure в линейный factor
    def noise_factor_linear(self):
        return self.db_to_linear(self.noise_figure)

    #  Реализует формулу: kTBF
    def noise_power_watt(self):
        return self.BOLTZMANN * self.temperature * self.bandwidth * self.noise_factor_linear()

    def noise_power_dbm(self):
        return self.watt_to_dbm(self.noise_power_watt())

    # Измерение мощности сигнала через среднее от квадрата модуля
    def measure_signal_power_watt(self, x):
        x = self.as_complex_array(x)
        if x.size == 0:
            return 0.0
        return float(np.mean(np.abs(x) ** 2))

    def measure_signal_power_dbm(self, x):
        return self.watt_to_dbm(self.measure_signal_power_watt(x))

    def symbol_rate_hz(self):
        if self.samples_per_symbol <= 0:
            return None
        return self.sample_rate / self.samples_per_symbol

    def bit_rate_hz(self):
        symbol_rate_hz = self.symbol_rate_hz()
        if symbol_rate_hz is None:
            return None
        return symbol_rate_hz * self.bits_per_symbol * self.coding_rate

    # Вычисление отношение мощностей сигнала и шума
    def compute_snr_db(self, signal_power_watt, noise_power_watt):
        if signal_power_watt <= 0 or noise_power_watt <= 0:
            return -np.inf
        return self.linear_to_db(signal_power_watt / noise_power_watt)

    # Вычисление carrier-to-noise в dB как разность dBm
    def compute_cn_db(self, signal_power_dbm, noise_power_dbm):
        return float(signal_power_dbm) - float(noise_power_dbm)

    # Использует связь между SNR и `Eb/N0`: Eb/N0[dB] = SNR[dB] + 10lg(B/Rb)
    def compute_ebn0_db(self, snr_db):
        bit_rate_hz = self._validate_positive(self.bit_rate_hz(), "bit_rate_hz")
        bandwidth_hz = self._validate_positive(self.bandwidth, "bandwidth_hz")
        return float(snr_db) + self.linear_to_db(bandwidth_hz / bit_rate_hz)

    # Возвращает полную дисперсию комплексного шума на sample
    def noise_variance_per_sample(self):
        return self.noise_power_watt()

    # Делит эту дисперсию пополам между I и Q
    def noise_std_per_dimension(self):
        return np.sqrt(self.noise_variance_per_sample() / 2.0)

    # Генерирует шум как сумму двух независимых нормальных последовательносте
    def generate_complex_noise(self, x_shape):
        sigma = self.noise_std_per_dimension()
        return sigma * (np.random.randn(*x_shape) + 1j * np.random.randn(*x_shape))

    def noise_metadata(self):
        return {
            "noise_model": "complex_awgn_sample_domain",
            "physical_channel_bandwidth_hz": self.physical_channel_bandwidth,
            "sample_rate_hz": self.sample_rate,
            "noise_bandwidth_hz": self.bandwidth,
            "noise_bandwidth_mode": self.noise_bandwidth_mode,
            "configured_bandwidth_hz": self.configured_bandwidth,
            "ebn0_interpretation": "sample_domain_effective_ebn0",
            "noise_bandwidth_explanation": (
                "Noise is added directly in the oversampled complex waveform, "
                "so the effective noise bandwidth equals sample_rate_hz."
                if self.noise_bandwidth_mode.lower() == "sample_rate"
                else "Noise bandwidth follows the selected receiver/physical bandwidth mode."
            ),
            "noise_figure_db": self.noise_figure,
            "temperature_k": self.temperature,
        }


class AWGNChannel(_PowerMath):

    def __init__(
        self,
        signal_power = None,
        temperature = 290.0,
        bandwidth = 200e3,
        physical_channel_bandwidth = None,
        noise_bandwidth_mode = "sample_rate",
        noise_figure = 7.0,
        sample_rate = 1.0,
        samples_per_symbol = 1.0,
        bits_per_symbol = 1.0,
        coding_rate = 1.0,
    ):
        super().__init__(
            temperature = temperature,
            bandwidth = bandwidth,
            physical_channel_bandwidth = physical_channel_bandwidth,
            noise_bandwidth_mode = noise_bandwidth_mode,
            noise_figure = noise_figure,
            sample_rate = sample_rate,
            samples_per_symbol = samples_per_symbol,
            bits_per_symbol = bits_per_symbol,
            coding_rate = coding_rate,
        )
        self.signal_power = signal_power # запоминает целевую мощность сигнала

    @classmethod
    def normalize_axis_metric(cls, axis_metric):
        normalized = "dbm" if axis_metric is None else str(axis_metric).lower()
        aliases = {
            "dbm": "dbm",
            "power": "dbm",
            "power_dbm": "dbm",
            "signal_power_dbm": "dbm",
            "prx": "dbm",
            "db": "snr_db",
            "snr": "snr_db",
            "snr_db": "snr_db",
            "ebn0": "ebn0_db",
            "ebn0_db": "ebn0_db",
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported axis metric: {axis_metric}")
        return aliases[normalized]

    def process(self, x):
        x = self.as_complex_array(x)
        noise_power_watt = self.noise_power_watt()      # cчитает ожидаемую мощность шума
        noise = self.generate_complex_noise(x.shape)    # создаёт случайную реализацию AWGN
        return x + noise, noise_power_watt              # возвращает зашумлённый сигнал и значение шумовой мощности
