import numpy as np
from pathlib import Path
import json
import matplotlib as plt
from scipy.stats import norm

class SoftGenerator:
    def __init__(self, channel_type="TCHFS", channel_model="awgn", profile="TU", llr_scale=8.0, is_working=True):
        self.llr_scale = llr_scale
        self.is_working = is_working
        

        if channel_type in ["CS1", "TCHFS", "MCS1"]:
            self.bits_per_symbol = 1
        elif channel_type in ["MCS5"]: 
            self.bits_per_symbol = 3
        else:
            raise ValueError(f"Неизвестный channel_type: {channel_type}. "
                           "Укажите bits_per_symbol вручную или добавьте в маппинг.")
        
        if channel_type in ["CS1", "TCHFS", "MCS1"]:
            if channel_model == "awgn":  
                dataset_path = Path(__file__).resolve().parent / "soft_data" / "gsm_omni_sector_pdtch_cs1_awgn.json"
                
            elif channel_model in ["rayleigh_single", "rayleigh_multipath"]and profile == "TU":
                dataset_path = Path(__file__).resolve().parent / "soft_data" / "gsm_omni_sector_pdtch_cs1_tu50.json"
                
            elif channel_model in ["rayleigh_single", "rayleigh_multipath"] and profile == "RA":
                dataset_path = Path(__file__).resolve().parent / "soft_data" / "gsm_omni_sector_pdtch_cs1_ra130.json"
                
            elif channel_model in ["rayleigh_single", "rayleigh_multipath"] and profile == "HT":
                dataset_path = Path(__file__).resolve().parent / "soft_data" / "gsm_omni_sector_pdtch_cs1_ht100.json"
                
        elif channel_type in ["MCS5"]:
            if channel_model == "awgn":  
                dataset_path = Path(__file__).resolve().parent / "soft_data" / "gsm_omni_sector_pdtch_mcs5_awgn.json"
                
            elif channel_model in ["rayleigh_single", "rayleigh_multipath"] and profile == "TU":
                dataset_path = Path(__file__).resolve().parent / "soft_data" / "gsm_omni_sector_pdtch_mcs5_tu50.json"
                
            elif channel_model in ["rayleigh_single", "rayleigh_multipath"] and profile == "RA":
                dataset_path = Path(__file__).resolve().parent / "soft_data" / "gsm_omni_sector_pdtch_mcs5_ra130.json"
                
            elif channel_model in ["rayleigh_single", "rayleigh_multipath"] and profile == "HT":
                dataset_path = Path(__file__).resolve().parent / "soft_data" / "gsm_omni_sector_pdtch_mcs5_ht100.json"
                    
        with open(dataset_path, "r") as f:
            data = json.load(f)

        data = sorted(data, key=lambda x: x["rssi"])
        data = data[:20]  

        ref_rssi = -119.0
        self.rssi_db = np.array([d["rssi"] for d in data]) + ref_rssi


        self.mean_abs = np.array([d["raw_llr_mean_abs"] for d in data])      # (N_points, bits_per_symbol)
        self.std      = np.array([d["raw_llr_std"]      for d in data])
        self.minv     = np.array([d["raw_llr_min"]      for d in data])
        self.maxv     = np.array([d["raw_llr_max"]      for d in data])
        self.uncoded_ber = np.array([d["uncoded_ber"]   for d in data])     # UB только одна

        if self.mean_abs.shape[1] != self.bits_per_symbol:
            raise ValueError(f"Несоответствие bits_per_symbol={self.bits_per_symbol} "
                           f"и данных в JSON ({self.mean_abs.shape[1]})")

    def get_uncoded_ber(self, rssi_list):
        return self.uncoded_ber[np.asarray(rssi_list).astype(int)]

    def get_soft_decisions(self, bits, rssi_list, num_sectors=2):
        bits = np.asarray(bits, dtype=np.int8).flatten()
        if bits.ndim != 1:
            raise ValueError("bits должен быть одномерным массивом")

        soft_list = []

        for rssi in np.asarray(rssi_list).flatten():
            sector_results = []
            for _ in range(num_sectors):
                result = self._generate_one_sector(bits, rssi)
                sector_results.append(result)
            soft_list.append(sector_results)

        return soft_list

    def _generate_one_sector(self, bits, rssi_idx):
        N = len(bits)
        idx = int(rssi_idx)

        sigma     = self.std[idx]          #  (bits_per_symbol,)
        minv_val  = self.minv[idx]
        maxv_val  = self.maxv[idx]
        ber       = self.uncoded_ber[idx]

        q_inv = norm.ppf(1 - ber)
        
        bit_pos = np.arange(N) % self.bits_per_symbol
        
        m = q_inv * sigma[bit_pos]                    
        llr_mean = m * (1 - 2 * bits)               

        llr = np.random.normal(loc=llr_mean, scale=sigma[bit_pos], size=N)
        llr = np.clip(llr, minv_val[bit_pos], maxv_val[bit_pos])
        llr = (np.clip(llr / self.llr_scale, -1.0, 1.0) * 127.0).astype(np.int8)
        
        return llr