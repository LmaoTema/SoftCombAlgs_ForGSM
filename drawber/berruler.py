import numpy as np
from datetime import datetime

class BERRuler:

    def __init__(
        self,
        axis_metric = None,
        sweep_mode = None,
        
        h2dB_init = 0.0,
        h2dB_init_step = 0.4,
        h2dB_min_step = 0.1,
        h2dB_max_step = 1.6,
        h2dB_max = 15,
        
        prx_dbm_init = -118.0,
        prx_dbm_init_step = 2.0,
        prx_dbm_min_step = 1.0,
        prx_dbm_max_step = 4.0,
        prx_dbm_max = -99.0,
        
        min_BER = 1e-6,
        min_FER = 1,
        min_NumErBits = 200,
        min_NumErFrames = 500,
        min_NumTrFrames = 500,
        max_NumTrBits = 1e8,
        max_NumTrFrames = 1e3,
        max_BERRate = 5,
        min_BERRate = 2,
        log_language = "Russian",
        enable_log = True,
        channel_type = "TCHFS",
        stop_by_min_BER = True,
        **kwargs,
    ):
        self.axis_metric = self._normalize_axis_metric(axis_metric if axis_metric is not None else sweep_mode)
        
        self.h2dB = h2dB_init       
        self.h2dB_step = h2dB_init_step
        self.h2dB_min_step = h2dB_min_step
        self.h2dB_max_step = h2dB_max_step
        self.h2dB_max = h2dB_max
        
        self.prx_dbm = prx_dbm_init
        self.prx_dbm_step = prx_dbm_init_step
        self.prx_dbm_min_step = prx_dbm_min_step
        self.prx_dbm_max_step = prx_dbm_max_step
        self.prx_dbm_max = prx_dbm_max
        self.x_values = []

        self.MinBER = min_BER
        self.MinFER = min_FER
        self.MinNumErBits = min_NumErBits
        self.MinNumErFrames = min_NumErFrames
        self.MinNumTrFrames = min_NumTrFrames
        self.MaxNumTrBits = max_NumTrBits
        self.MaxNumTrFrames = max_NumTrFrames
        self.MaxBERRate = max_BERRate
        self.MinBERRate = min_BERRate

        self.enable_log = enable_log
        self.log_language = log_language
        self.channel_type = channel_type
        self.stop_by_min_BER = stop_by_min_BER

        tmp_blocks = self._slice_blocks(np.zeros(1)) 
        self.stats = {
            name: {
                "NumTrBits": 0, 
                "NumErBits": 0, 
                "NumTrFrames": 0, 
                "NumErFrames": 0
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
            "average_channel_power": [],
            "applied_signal_power_dbm": [],
            "measured_signal_power_dbm": [],
            "noise_power_dbm": [],
            "noise_variance_per_sample": [],
            "snr_db": [],
            "ebn0_db": [],
            "carrier_to_noise_db": [],
            "symbol_rate_hz": [],
            "bit_rate_hz": [],
            "outage_rate": [],
        }
        self._channel_metric_accumulator = {
            "average_channel_power": [],
            "applied_signal_power_dbm": [],
            "measured_signal_power_dbm": [],
            "noise_power_dbm": [],
            "noise_variance_per_sample": [],
            "snr_db": [],
            "ebn0_db": [],
            "carrier_to_noise_db": [],
            "symbol_rate_hz": [],
            "bit_rate_hz": [],
            "outage": [],
            "axis_values": [],
        }
        
        self.isStop = False

    @staticmethod
    def _normalize_axis_metric(axis_metric):
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
        return aliases.get(normalized, "dbm")

    def _slice_blocks(self, bits):
        if self.channel_type in {"UNCODED", "RAW", "FULL"}:
            return {"full": bits[:]}
        if self.channel_type == "TCHFS":
            return {
                "class1": bits[0:182],
                "class2": bits[182:260],
                "full": bits[0:260],
            }
        if self.channel_type == "CS1":
            return {"full": bits[:]}
        if self.channel_type == "MCS1":
            return {
                "header": bits[0:80],
                "data": bits[80:456],
            }
        if self.channel_type == "MCS5":
            return {
                "header": bits[0:136],
                "data": bits[136:1384],
            }
        return {"full": bits[:]}


    def update_frame(self, tx_bits, rx_bits, channel_output = None):
        tx_blocks = self._slice_blocks(tx_bits)
        rx_blocks = self._slice_blocks(rx_bits)

        for name, tx_blk in tx_blocks.items():
            rx_blk = rx_blocks[name]
            n_bits = len(tx_blk)
            n_errors = np.sum(np.array(tx_blk) != np.array(rx_blk))

            self.stats[name]["NumTrBits"] += n_bits
            self.stats[name]["NumTrFrames"] += 1
            self.stats[name]["NumErBits"] += n_errors
            if n_errors > 0:
                self.stats[name]["NumErFrames"] += 1

        if channel_output is None:
            return

        if channel_output.average_channel_power is not None:
            self._channel_metric_accumulator["average_channel_power"].append(channel_output.average_channel_power)
        if channel_output.applied_signal_power_dbm is not None:
            self._channel_metric_accumulator["applied_signal_power_dbm"].append(channel_output.applied_signal_power_dbm)
        if channel_output.measured_signal_power_dbm is not None:
            self._channel_metric_accumulator["measured_signal_power_dbm"].append(channel_output.measured_signal_power_dbm)
        if channel_output.noise_power_dbm is not None:
            self._channel_metric_accumulator["noise_power_dbm"].append(channel_output.noise_power_dbm)
        if channel_output.noise_variance_per_sample is not None:
            self._channel_metric_accumulator["noise_variance_per_sample"].append(channel_output.noise_variance_per_sample)
        if channel_output.snr_db is not None:
            self._channel_metric_accumulator["snr_db"].append(channel_output.snr_db)
        if channel_output.ebn0_db is not None:
            self._channel_metric_accumulator["ebn0_db"].append(channel_output.ebn0_db)
        if channel_output.carrier_to_noise_db is not None:
            self._channel_metric_accumulator["carrier_to_noise_db"].append(channel_output.carrier_to_noise_db)
        if channel_output.symbol_rate_hz is not None:
            self._channel_metric_accumulator["symbol_rate_hz"].append(channel_output.symbol_rate_hz)
        if channel_output.bit_rate_hz is not None:
            self._channel_metric_accumulator["bit_rate_hz"].append(channel_output.bit_rate_hz)
        self._channel_metric_accumulator["outage"].append(bool(channel_output.outage))
        axis_value = channel_output.metadata.get("axis_value")
        if axis_value is not None:
            self._channel_metric_accumulator["axis_values"].append(float(axis_value))
            
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
    
    def _append_channel_metric(self, key, default = None):
        values = self._channel_metric_accumulator[key]
        if not values:
            if key == "outage":
                self.channel_metrics["outage_rate"].append(default)
            else:
                self.channel_metrics[key].append(default)
            return
        if key == "outage":
            self.channel_metrics["outage_rate"].append(float(np.mean(values)))
            return
        self.channel_metrics[key].append(float(np.mean(values)))

    def _current_axis_label(self, x_value):
        if self.axis_metric == "dbm":
            return f"P_rx={x_value:.2f} dBm"
        if self.axis_metric == "ebn0_db":
            return f"Eb/N0={x_value:.2f} dB"
        return f"SNR={x_value:.2f} dB"

    def finalize_point(self):
        axis_values = self._channel_metric_accumulator["axis_values"]
        x_value = self.prx_dbm if not axis_values else float(np.mean(axis_values))
        self.x_values.append(x_value)
        x_label = self._current_axis_label(x_value)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for name, stat in self.stats.items():
            n_bits = max(1, stat["NumTrBits"])
            n_err_bits = stat["NumErBits"]
            ber = n_err_bits / n_bits if n_err_bits > 0 else 3.0 / n_bits

            n_frames = max(1, stat["NumTrFrames"])
            n_err_frames = stat["NumErFrames"]

            
            fer = n_err_frames / n_frames if n_err_frames > 0 else 3.0 / n_frames
           
            self.results[name]["BER"].append(ber)
            self.results[name]["FER"].append(fer)

            if self.enable_log:
                print(f"{ts} | {x_label} | {name} BER={ber:.3e} | FER={fer:.3e} | Frames={stat['NumTrFrames']}")
        
        self._append_channel_metric("average_channel_power", default = None)
        self._append_channel_metric("applied_signal_power_dbm", default = None)
        self._append_channel_metric("measured_signal_power_dbm", default = None)
        self._append_channel_metric("noise_power_dbm", default = None)
        self._append_channel_metric("noise_variance_per_sample", default = None)
        self._append_channel_metric("snr_db", default = None)
        self._append_channel_metric("ebn0_db", default = None)
        self._append_channel_metric("carrier_to_noise_db", default = None)
        self._append_channel_metric("symbol_rate_hz", default = None)
        self._append_channel_metric("bit_rate_hz", default = None)
        self._append_channel_metric("outage", default = 0.0)
        
        main_ber = list(self.results.values())[0]["BER"][-1]
        if self.stop_by_min_BER and main_ber < self.MinBER:
            self.isStop = True

        self._update_power_step()
        self.reset()
        if self.enable_log:
            print()

    def _update_power_step(self):
        if len(self.x_values) < 2:
            self.prx_dbm += self.prx_dbm_step
            return

        ber_prev = list(self.results.values())[0]["BER"][-2]
        ber_curr = list(self.results.values())[0]["BER"][-1]
        rate = np.inf if ber_curr == 0 else ber_prev / ber_curr
        
        step = self.prx_dbm_step
        min_step = self.prx_dbm_min_step
        max_step = self.prx_dbm_max_step
            
        if rate > self.MaxBERRate:
            dec = (
                0.5 if rate / self.MaxBERRate < 4 else
                0.25 if rate / self.MaxBERRate < 16 else 
                0.125 if rate / self.MaxBERRate < 64 else 
                0.0625
            )
            step = max(step * dec, min_step)
        elif rate < self.MinBERRate:
            step = min(step * 2, max_step)
            self.prx_dbm_step = step
            self.prx_dbm += self.prx_dbm_step
            if self.prx_dbm > self.prx_dbm_max:
                self.isStop = True

    def reset(self):
        for stat in self.stats.values():
            stat["NumTrBits"] = 0
            stat["NumErBits"] = 0
            stat["NumTrFrames"] = 0
            stat["NumErFrames"] = 0
        for key in self._channel_metric_accumulator:
            self._channel_metric_accumulator[key] = []
    
    def get_results(self):
        return {
            "axis_metric": self.axis_metric,
            "x": np.array(self.x_values),
            "results": self.results,
            "channel_metrics": self.channel_metrics,
        }