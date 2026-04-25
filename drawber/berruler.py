import numpy as np
from datetime import datetime

class BERRuler:

    def __init__(
        self,
        sweep_mode = "snr",
        
        h2dB_init = 0.0,
        h2dB_init_step = 0.4,
        h2dB_min_step = 0.1,
        h2dB_max_step = 1.6,
        h2dB_max = 15,
        
        prx_dbm_init = -118.0,
        prx_dbm_init_step = 2.0,
        prx_dbm_min_step = 0.5,
        prx_dbm_max_step =4.0,
        prx_dbm_max = -90.0,
        
        min_BER = 1e-4,
        min_FER = 1,
        min_NumErBits = 200,
        min_NumErFrames = 500,
        min_NumTrFrames = 500,
        max_NumTrBits = 1e8,
        max_NumTrFrames = np.inf,
        max_BERRate = 5,
        min_BERRate = 2,
        log_language = "Russian",
        enable_log = True,
        channel_type = "TCHFS",
        stop_by_min_BER = True,
        **kwargs
    ):
        self.sweep_mode = sweep_mode
        
        self.h2dB = h2dB_init
        self.h2dB_step = h2dB_init_step
        self.h2dB_min_step = h2dB_min_step
        self.h2dB_max_step = h2dB_max_step
        self.h2dB_max = h2dB_max
        self.h2dBs = []
        
        self.prx_dbm = prx_dbm_init
        self.prx_dbm_step = prx_dbm_init_step
        self.prx_dbm_min_step = prx_dbm_min_step
        self.prx_dbm_max_step = prx_dbm_max_step
        self.prx_dbm_max = prx_dbm_max
        self.prx_dbms = []

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
                "NumTrBits":0, 
                "NumErBits":0, 
                "NumTrFrames":0, 
                "NumErFrames":0
            } 
            for name in tmp_blocks
        }
        self.results = {
            name: {
                "BER":[], 
                "FER":[]
            } 
            for name in tmp_blocks
        }

        self.isStop = False

    def _slice_blocks(self, bits):
        if self.channel_type == "TCHFS":
            return {
                "class1": bits[0:182],
                "class2": bits[182:260],
                "full": bits[0:260]
            }
        elif self.channel_type == "CS1":
            return {"full": bits[:]}
        elif self.channel_type == "MCS1":
            return {
                "header": bits[0:80],
                "data": bits[80:456]
            }
        elif self.channel_type == "MCS5":
            return {
                "header": bits[0:136],
                "data": bits[136:1384]
            }
        else:
            return {"full": bits[:]}


    def update_frame(self, tx_bits, rx_bits):
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


    def is_point_finished(self):
        total_tr_bits = sum(s["NumTrBits"] for s in self.stats.values())
        total_er_bits = sum(s["NumErBits"] for s in self.stats.values())
        total_tr_frames = sum(s["NumTrFrames"] for s in self.stats.values())
        total_er_frames = sum(s["NumErFrames"] for s in self.stats.values())

        enough_statistics = (
            total_er_bits >= self.MinNumErBits and
            total_er_frames >= self.MinNumErFrames and
            total_tr_frames >= self.MinNumTrFrames
        )
        complexity_exceeded = (
            total_tr_bits >= self.MaxNumTrBits or
            total_tr_frames >= self.MaxNumTrFrames
        )
        return enough_statistics or complexity_exceeded
    
    def get_current_x(self):
        return self.h2dB if self.sweep_mode == "snr" else self.prx_dbm

    def finalize_point(self):
        x = self.get_current_x()

        if self.sweep_mode == "snr":
            self.h2dBs.append(x)
            x_label = f"h2={x:.2f} dB"
        else:
            self.prx_dbms.append(x)
            x_label = f"P_rx={x:.2f} dBm"

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for name, stat in self.stats.items():
            n_bits = max(1, stat["NumTrBits"])
            n_err_bits = stat["NumErBits"]

            if n_err_bits > 0:
                ber = n_err_bits / n_bits
            else:
                ber = 3.0 / n_bits

            n_frames = max(1, stat["NumTrFrames"])
            n_err_frames = stat["NumErFrames"]

            if n_err_frames > 0:
                fer = n_err_frames / n_frames
            else:
                fer = 3.0 / n_frames
           
            self.results[name]["BER"].append(ber)
            self.results[name]["FER"].append(fer)

            if self.enable_log:
                print(f"{ts} | {x_label} | {name} BER={ber:.3e} | FER={fer:.3e} | Frames={stat['NumTrFrames']}")

        main_ber = list(self.results.values())[0]["BER"][-1]
        if self.stop_by_min_BER and main_ber < self.MinBER:
            self.isStop = True

        self._update_snr_step()
        self.reset()
        if self.enable_log:
            print()

    def _update_snr_step(self):
        x_values = self.h2dBs if self.sweep_mode == "snr" else self.prx_dbms
        
        if len(x_values) < 2:
            if self.sweep_mode == "snr":
                self.h2dB += self.h2dB_step
            else:
                self.prx_dbm += self.prx_dbm_step
            return

        ber_prev = list(self.results.values())[0]["BER"][-2]
        ber_curr = list(self.results.values())[0]["BER"][-1]
        rate = np.inf if ber_curr == 0 else ber_prev / ber_curr
        
        if self.sweep_mode == "snr":
            step = self.h2dB_step
            min_step = self.h2dB_min_step
            max_step = self.h2dB_max_step
        else:
            step = self.prx_dbm_step
            min_step = self.prx_dbm_min_step
            max_step = self.prx_dbm_max_step
            
        if rate > self.MaxBERRate:
            dec = (
                0.5 if rate/self.MaxBERRate < 4 else
                0.25 if rate/self.MaxBERRate < 16 else 
                0.125 if rate/self.MaxBERRate < 64 else 
                0.0625
            )
            step = max(step * dec, min_step)
            
        elif rate < self.MinBERRate:
            step = min(step * 2, max_step)

        if self.sweep_mode == "snr":
            self.h2dB_step = step
            self.h2dB += self.h2dB_step
            if self.h2dB > self.h2dB_max:
                self.isStop = True
        else:
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

    # Получение результатов в виде словаря для графиков
    def get_results(self):
        return {
            "sweep_mode": self.sweep_mode,
            "x": np.array(self.h2dBs if self.sweep_mode == "snr" else self.prx_dbms),
            "results": self.results
        }