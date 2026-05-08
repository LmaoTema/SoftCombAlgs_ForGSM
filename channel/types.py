from dataclasses import dataclass, field
from typing import Any

import numpy as np

# структура данных с описанием канала

@dataclass
class ChannelState:
    kind: str       # тип канала
    sample_rate: float | None = None        
    samples_per_symbol: float | None = None # число отсчетов на символ
    symbol_energy: float | None = None      # энергия символа
    average_sample_power: float | None = None   # cредняя мощность выходного дискретного сигнала
    average_channel_power: float | None = None  # cредняя мощность самого канала
    impulse_response: Any = None    # ИХ
    flat_gain: Any = None           # последовательность `h[n]`
    path_delays_samples: Any = None # задержки
    path_powers: Any = None         # мощности лучей 
    path_gains: Any = None          # временные коэффициенты лучей
    metadata: dict[str, Any] = field(default_factory = dict)


@dataclass
class ChannelOutput:
    signal: np.ndarray      # массив отсчётов сигнала после канала
    channel_state: ChannelState | None = None   # состояние канала
    noise_power_watt: float | None = None       # мощность шума в ваттах
    noise_power_dbm: float | None = None        # мощность шума в dBm
    noise_variance_per_sample: float | None = None  # дисперсия шума на комплексный отсчёт
    applied_signal_power_watt: float | None = None  # какая целевая мощность сигнала была задана после large-scale scaling
    applied_signal_power_dbm: float | None = None   #
    measured_signal_power_watt: float | None = None # какая мощность реально получилась после scaling
    measured_signal_power_dbm: float | None = None  #
    measured_output_power_watt: float | None = None # какая мощность получилась уже после AWGN
    measured_output_power_dbm: float | None = None  #
    normalized_signal_power_watt: float | None = None   # какая мощность у финального выхода после unit normalization
    normalization_gain: float | None = None             # на какой коэффициент был умножен сигнал при финальной нормировке
    snr_db: float | None = None     
    ebn0_db: float | None = None    
    carrier_to_noise_db: float | None = None    
    symbol_rate_hz: float | None = None         
    bit_rate_hz: float | None = None            
    measured_sample_snr_db: float | None = None  
    average_channel_power: float | None = None  # средняя мощность канала
    outage: bool = False    #
    metadata: dict[str, Any] = field(default_factory = dict)
