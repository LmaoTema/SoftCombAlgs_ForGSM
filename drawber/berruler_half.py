import numpy as np
from datetime import datetime

class HalfBERRuler:
    def __init__(
        self,
        rssi_points,
        enable_log=True,
        channel_type="TCHFS",
        **kwargs
    ):
        self.enable_log = enable_log
        self.isStop = False
        self.rssi_points = np.asarray(rssi_points)
        self.point_index = 0
        self.channel_type = channel_type.upper() 

        self.stop_by_min_BER = kwargs.get("stop_by_min_BER", False)
        self.min_BER = kwargs.get("min_BER", 1e-3)
        self.MinNumErBits = kwargs.get("min_NumErBits", 800)
        self.MinNumTrFrames = kwargs.get("min_NumTrFrames", 10000)
        self.MaxNumTrBits = kwargs.get("max_NumTrBits", 5e6)
        self.MinNumErFrames = kwargs.get("min_NumErFrames", 500)

        self.x_values = []


        tmp_blocks = self._slice_blocks(np.zeros(1, dtype=int))
        self.stats = {
            name: {
                "NumTrBits": 0,
                "NumErBits": 0,
                "NumTrFrames": 0,
                "NumErFrames": 0,
            }
            for name in tmp_blocks
        }

        self.results = {
            name: {
                "BER": [],
                "FER": [],
            }
            for name in tmp_blocks
        }

        self.current_uncoded_ber = None
        self._reset_accumulators()

    def _slice_blocks(self, bits):
        bits = np.asarray(bits).reshape(-1)
        if self.channel_type in {"UNCODED", "RAW", "FULL", "CS1"}:
            return {"full": bits[:]}
        
        if self.channel_type == "TCHFS":
            return {
                "class1": bits[0:182],
                "class2": bits[182:260],
                "full": bits[0:260],
            }
        
        if self.channel_type == "MCS1":
            return {
                "header": bits[0:31],
                "data": bits[31:209],
                "full": bits[0:209]
            }
        
        if self.channel_type == "MCS5":
            return {
                "header": bits[0:136],
                "data": bits[136:1384],
                "full": bits[0:1384]
            }
        return {"full": bits[:]}

    @property
    def current_x(self):
        idx = min(self.point_index, len(self.rssi_points) - 1)
        return float(self.rssi_points[idx])

    def update_frame(self, tx_bits, rx_bits, uncoded_ber=None):
        tx_blocks = self._slice_blocks(tx_bits)
        rx_blocks = self._slice_blocks(rx_bits)

        for name in tx_blocks:
            tx_blk = tx_blocks[name]
            rx_blk = rx_blocks[name]

            n_bits = len(tx_blk)
            n_errors = np.sum(tx_blk != rx_blk)

            self.stats[name]["NumTrBits"] += n_bits
            self.stats[name]["NumErBits"] += n_errors
            self.stats[name]["NumTrFrames"] += 1
            if n_errors > 0:
                self.stats[name]["NumErFrames"] += 1

        if uncoded_ber is not None:
            self.current_uncoded_ber = uncoded_ber

    def is_point_finished(self):
        total_er_bits = sum(s["NumErBits"] for s in self.stats.values())
        total_tr_frames = sum(s["NumTrFrames"] for s in self.stats.values())
        total_tr_bits = sum(s["NumTrBits"] for s in self.stats.values())

        enough_errors = total_er_bits >= self.MinNumErBits
        enough_frames = total_tr_frames >= self.MinNumTrFrames
        max_bits = total_tr_bits >= self.MaxNumTrBits

        return enough_errors or enough_frames or max_bits

    def finalize_point(self):
        rssi = self.current_x
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.x_values.append(rssi)

        for name, stat in self.stats.items():
            n_bits = max(1, stat["NumTrBits"])
            ber = stat["NumErBits"] / n_bits

            n_frames = max(1, stat["NumTrFrames"])
            fer = stat["NumErFrames"] / n_frames

            self.results[name]["BER"].append(ber)
            self.results[name]["FER"].append(fer)

            if self.enable_log:
                uncoded_str = f"{self.current_uncoded_ber:.3e}" if self.current_uncoded_ber is not None else "N/A"
                print(f"{ts} | RSSI={rssi:6.2f} | {name:>8} | "
                      f"BER={ber:.3e} | FER={fer:.3e} | "
                      f"Uncoded={uncoded_str} | "
                      f"bits={stat['NumTrBits']:,} | errs={stat['NumErBits']:,}")

        if self.enable_log:
            print()
        if self.stop_by_min_BER:
            main_ber = list(self.results.values())[0]["BER"][-1]
            if main_ber <= self.min_BER:
                self.isStop = True

        self._reset_accumulators()
        self.point_index += 1

    def _reset_accumulators(self):
        for stat in self.stats.values():
            stat["NumTrBits"] = 0
            stat["NumErBits"] = 0
            stat["NumTrFrames"] = 0
            stat["NumErFrames"] = 0
        self.current_uncoded_ber = None

    def get_results(self):
        return {
            "x": np.array(self.x_values),
            "results": self.results,
            "axis_metric": "dBm"
        }