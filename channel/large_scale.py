import numpy as np

# получает сигнал и приводит его среднюю мощность к заданному
class ReceivedPowerScaler:

    @staticmethod
    def dbm_to_watt(p_dbm):
        return 10 ** ((float(p_dbm) - 30.0) / 10.0)

    @staticmethod
    def watt_to_dbm(p_watt):
        if p_watt <= 0:
            return -np.inf
        return 10.0 * np.log10(float(p_watt)) + 30.0

    @staticmethod
    def measure_signal_power_watt(x):
        x = np.asarray(x, dtype = np.complex128)
        if x.size == 0:
            return 0.0
        return float(np.mean(np.abs(x) ** 2))

    def process(self, x, signal_power_dbm):
        x = np.asarray(x, dtype = np.complex128)
        current_power = self.measure_signal_power_watt(x) # измеряем текущую среднюю мощность
        if current_power <= 0:
            raise ValueError("Signal power is zero, cannot scale")

        target_power_watt = self.dbm_to_watt(signal_power_dbm)  # переводим заданный уровень из dBm в ватт
        scale = np.sqrt(target_power_watt / current_power)      # корень из отношения мощностей
        y = x * scale                                           # чтобы изменить мощность, нужно умножить амплитуду на корень
        measured_power_watt = self.measure_signal_power_watt(y) # записываем, что получилось
        metadata = {
            "input_power_watt": current_power,
            "target_power_watt": target_power_watt,
            "applied_scale_linear": float(scale),
            "output_power_watt": measured_power_watt,
        }
        return y, target_power_watt, measured_power_watt, metadata  # Возвращаем: масштабированный сигнал; целевую мощность; измеренную мощность; diagnostic metadata
