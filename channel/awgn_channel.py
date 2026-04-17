import numpy as np


class AWGNChannel():
    BOLTZMANN = 1.380649e-23    # [Дж/К]

    def __init__(
        self, 
        snr_db = None, 
        temperature = 290.0, 
        bandwidth = 200e3, 
        noise_figure = 7.0, 
        signal_power = None,
        ):
        
        self.snr_db = snr_db
        self.temperature = temperature
        self.bandwidth = bandwidth
        self.noise_figure = noise_figure
        self.signal_power = signal_power

    @staticmethod
    def db_to_linear(x_db):
        return 10 ** (x_db / 10.0)
    
    @staticmethod
    def linear_to_do(x_lin):
        if x_lin <= 0:
            return -np.inf
        return 10.0 * np.log10(x_lin)
    
    @staticmethod
    def dbm_to_watt(p_dbm):
        return 10 ** ((p_dbm - 30.0) / 10.0)
    
    @staticmethod
    def watt_to_dbm(p_watt):
        if p_watt <= 0:
            return -np.inf
        return 10.0 * np.log10(p_watt) + 30.0
    
    def get_noise_power_watt(self):
        noise_factor = self.db_to_linear(self.noise_figure)
        return (
            self.BOLTZMANN
            * self.temperature
            * self.bandwidth
            * noise_factor
        )
    
    def get_noise_power_dbm(self):
        return self.watt_to_dbm(self.get_noise_power_watt())
    
    def scale_signal_to_power(self, x, target_power_dbm):
        x = np.asarray(x, dtype = np.complex128)
        
        current_power = np.mean(np.abs(x) ** 2)
        if current_power <= 0:
            raise ValueError("Signal power is zero, cannot scale")
        
        target_power_watt = self.dbm_to_watt(float(target_power_dbm))
        scale = np.sqrt(target_power_watt / current_power)
        
        return x * scale
    
    def _process_prx_mode(self, x, signal_power = None):
        x = np.asarray(x, dtype = np.complex128)
        
        target_power_dbm = signal_power if signal_power is not None else self.signal_power
        if target_power_dbm is None:
            raise ValueError("For P_rx mode, signal_power must be provided.")
        
        x = self.scale_signal_to_power(x, target_power_dbm)

        noise_power = self.get_noise_power_watt()
        noise = np.sqrt(noise_power / 2.0) * (
            np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
        )

        return x + noise

    def _process_snr_mode(self, x, snr_db = None):
        x = np.asarray(x, dtype = np.complex128)
       
        snr_db_used = snr_db if snr_db is not None else self.snr_db
        if snr_db_used is None:
            raise ValueError(
                "For SNR mode, snr_db must be provided."
            )

        noise_power_watt = self.get_noise_power_watt()
        noise_power_dbm = self.watt_to_dbm(noise_power_watt)
        
        target_signal_power_dbm = noise_power_dbm + float(snr_db_used)
        
        x = self.scale_signal_to_power(x, target_signal_power_dbm)
        
        noise = np.sqrt(noise_power_watt / 2.0) * (
            np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
        )
        
        return x + noise
        
    def process(self, x, signal_power = None, snr_db = None):
        x = np.asarray(x, dtype = np.complex128)

        use_prx_mode = ((signal_power is not None) 
            or (
                signal_power is None 
                and snr_db is None 
                and self.signal_power is not None
            )
        )
        
        if use_prx_mode:
            return self._process_prx_mode(x, signal_power = signal_power)

        return self._process_snr_mode(x, snr_db = snr_db)