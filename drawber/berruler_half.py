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
        self.MaxNumTrFrames = kwargs.get("max_NumTrFrames", 500)
        self.prx_dbm_max = kwargs.get("prx_dbm_max", -90.0)
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

        self.channel_metrics = {
            "ci_db": []
        }
        
        self._channel_metric_accumulator = {
            "ci_db": [],
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
                "header": bits[:37],
                "data": bits[37:],
                "full": bits[:]
            }
        return {"full": bits[:]}

    @property
    def current_x(self):
        idx = min(self.point_index, len(self.rssi_points) - 1)
        return float(self.rssi_points[idx])

    def update_frame(self, tx_bits, rx_bits, uncoded_ber=None, channel_output=None):
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
            
        if channel_output is not None:
            self._channel_metric_accumulator["ci_db"].append(float(channel_output))
    
    
    def _append_channel_metric(self, key, default=None):
        values = self._channel_metric_accumulator.get(key, [])
        if not values:
            self.channel_metrics[key].append(default)
            return
        
        if key == "ci_db":
            self.channel_metrics[key].append(float(np.mean(values)))
        else:
            self.channel_metrics[key].append(float(np.mean(values)))
    
    def is_point_finished(self):
        total_tr_bits = sum(s["NumTrBits"] for s in self.stats.values())
        total_er_bits = sum(s["NumErBits"] for s in self.stats.values())
        total_tr_frames = sum(s["NumTrFrames"] for s in self.stats.values())
        total_er_frames = sum(s["NumErFrames"] for s in self.stats.values())

        enough_statistics = (
            total_er_bits >= self.MinNumErBits
            and total_er_frames >= self.MinNumErFrames
            and total_tr_frames >= self.MinNumTrFrames
        )
        complexity_exceeded = (
            total_tr_bits >= self.MaxNumTrBits
            or total_tr_frames >= self.MaxNumTrFrames
        )
        return enough_statistics or complexity_exceeded

    def finalize_point(self):
        rssi = self.current_x
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.x_values.append(rssi)

        for name, stat in self.stats.items():
            n_bits = max(1, stat["NumTrBits"])
            n_err_bits = stat["NumErBits"]
            ber = n_err_bits / n_bits if n_err_bits > 0 else np.nan

            n_frames = max(1, stat["NumTrFrames"])
            n_err_frames = stat["NumErFrames"]

            fer = n_err_frames / n_frames if n_err_frames > 0 else np.nan

            if n_err_bits > 0:
                all_blocks_error_free = False
            
            self.results[name]["BER"].append(ber)
            self.results[name]["FER"].append(fer)
            
            if self.enable_log:
                ber_label = f"{ber:.3e}" if np.isfinite(ber) else "not-observed"
                fer_label = f"{fer:.3e}" if np.isfinite(fer) else "not-observed"
                # uncoded_str = f"{self.current_uncoded_ber:.3e}" if self.current_uncoded_ber is not None else "N/A"
                ci_str = f"{np.mean(self._channel_metric_accumulator['ci_db']):.2f}" if self._channel_metric_accumulator["ci_db"] else "N/A"
                print(f"{ts} | RSSI={rssi:6.2f} | {name:>8} | "
                      f"BER={ber_label} | FER={fer_label} | "
                      f"CI={ci_str} | "
                      f"Frames={stat['NumTrFrames']:,} | Errors={stat['NumErBits']:,}")
                
        self._append_channel_metric("ci_db", default=None)

        if self.enable_log:
            print()
        main_ber = list(self.results.values())[0]["BER"][-1]
        if self.stop_by_min_BER and np.isfinite(main_ber) and main_ber < self.min_BER:
            self.isStop = True
        
        if rssi > self.prx_dbm_max:
            self.isStop = True
                    
        if all_blocks_error_free:
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
        for key in self._channel_metric_accumulator:
            self._channel_metric_accumulator[key] = []

    def get_results(self):
        return {
            "x": np.array(self.x_values),
            "results": self.results,
            "axis_metric": "dBm",
            "channel_metrics": self.channel_metrics, 
        }