import numpy as np
from datetime import datetime


class HalfBERRuler:

    def __init__(
        self,
        rssi_points,
        enable_log=True,
        **kwargs
    ):  
        self.enable_log = enable_log
        self.isStop = False
        self.rssi_points = np.array(rssi_points)
        self.enable_log = enable_log
        self.stop_by_min_BER = kwargs.get("stop_by_min_BER", False)
        self.min_BER = kwargs.get("min_BER", 1e-3)
        self.point_index = 0
        self.MinNumErBits = kwargs.get("min_NumErBits", 500)
        self.MinNumTrFrames = kwargs.get("min_NumTrFrames", 1000)
        self.MaxNumTrBits = kwargs.get("max_NumTrBits", 5e6)
        self.MinNumErFrames = kwargs.get("min_NumErFrames", 500)    
        self.x_values = []

        self.results = {
            "coded": {
                "BER": [],
                "FER": [],
            },
            "uncoded": {
                "BER": [],
            }
        }

        self.current_uncoded_ber = None
        self.stats = {
        "NumTrBits": 0,
        "NumErBits": 0,
        "NumTrFrames": 0,
        "NumErFrames": 0,}

        self._reset_accumulators()

    @property
    def current_x(self):
        idx = min(self.point_index, len(self.rssi_points) - 1)
        return self.rssi_points[idx]


    def update_frame(self, tx_bits, rx_bits, uncoded_ber=None):

        tx_bits = np.asarray(tx_bits).reshape(-1)
        rx_bits = np.asarray(rx_bits).reshape(-1)

        self._num_bits += tx_bits.size
        self._num_errors += np.sum(tx_bits != rx_bits)

        self._num_frames += 1
        if np.any(tx_bits != rx_bits):
            self._num_err_frames += 1

        if uncoded_ber is not None:
            self.current_uncoded_ber = uncoded_ber

    def is_point_finished(self):

        enough_errors = (
            self.stats["NumErBits"] >= self.MinNumErBits
        )

        enough_frames = (
            self.stats["NumTrFrames"] >= self.MinNumTrFrames
        )

        max_bits = (
            self.stats["NumTrBits"] >= self.MaxNumTrBits
        )

        return enough_errors or enough_frames or max_bits
    
    def finalize_point(self):

        ber = self._num_errors / max(1, self._num_bits)
        fer = self._num_err_frames / max(1, self._num_frames)

        rssi = self.current_x

        self.x_values.append(rssi)

        self.results["coded"]["BER"].append(ber)
        self.results["coded"]["FER"].append(fer)

        self.results["uncoded"]["BER"].append(self.current_uncoded_ber)

        # лог
        if self.enable_log:
            print(f"RSSI={rssi:.2f} | BER={ber:.3e}")

        if self.stop_by_min_BER:
            if ber <= self.min_BER:
                self.isStop = True

        self._reset_accumulators()
        self.point_index += 1

    def _reset_accumulators(self):
        self._num_bits = 0
        self._num_errors = 0
        self._num_frames = 0
        self._num_err_frames = 0
        self.current_uncoded_ber = None
        
    def reset(self):

        self.stats["NumTrBits"] = 0
        self.stats["NumErBits"] = 0
        self.stats["NumTrFrames"] = 0
        self.stats["NumErFrames"] = 0

    def get_results(self):

        return {
            "x": np.array(self.x_values),
            "results": self.results,
        }