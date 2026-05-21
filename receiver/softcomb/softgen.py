import numpy as np
from pathlib import Path
import json


class SoftGenerator:
    def __init__(self, channel_type="TCHFS", channel_model="awgn", profile="TU", llr_scale=8.0, is_working=True):
        self.llr_scale = llr_scale
        self.is_working = is_working
        
        if channel_type in ["CS1", "TCHFS"]:
            if channel_model == "awgn":  
                dataset_path = Path(__file__).resolve().parent / "soft_data" / "gsm_omni_sector_pdtch_cs1_awgn.json"
                
            elif channel_model in ["rayleigh_single", "rayleigh_multipath"]and profile == "TU":
                dataset_path = Path(__file__).resolve().parent / "soft_data" / "gsm_omni_sector_pdtch_cs1_tu50.json"
                
            elif channel_model in ["rayleigh_single", "rayleigh_multipath"] and profile == "RA":
                dataset_path = Path(__file__).resolve().parent / "soft_data" / "gsm_omni_sector_pdtch_cs1_ra130.json"
                
            elif channel_model in ["rayleigh_single", "rayleigh_multipath"] and profile == "HT":
                dataset_path = Path(__file__).resolve().parent / "soft_data" / "gsm_omni_sector_pdtch_cs1_ht100.json"
                
        elif channel_type in ["MCS1", "MCS5"]:
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
        # в случае АБГШ обрезаем до 10 первых точек
        
        data = data[:10]
            
        ref_rssi = -108.0
        self.rssi_db = np.array([d["rssi"] for d in data]) + ref_rssi
        
        self.mean = np.array([d["raw_llr_mean_abs"][0] for d in data])
        self.std  = np.array([d["raw_llr_std"][0]  for d in data])
        self.minv = np.array([d["raw_llr_min"][0]  for d in data])
        self.maxv = np.array([d["raw_llr_max"][0]  for d in data])
        self.uncoded_ber = np.array([d["uncoded_ber"] for d in data])

    def get_uncoded_ber(self, rssi_list):
        return self.uncoded_ber[rssi_list]

    def get_soft_decisions(self, bits, rssi_list, num_sectors=2):

        bits = np.asarray(bits, dtype=np.int8)
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

    def _generate_one_sector(self, bits, rssi):
        
        N = len(bits)
        idx = rssi
        
        mu = self.mean[idx]
        sigma = self.std[idx]
        minv_val = self.minv[idx]
        maxv_val = self.maxv[idx]

        llr_mean = mu * (1 - 2*bits)

        llr = np.random.normal(loc=llr_mean, scale=sigma, size=N)
        llr = np.clip(llr, minv_val, maxv_val)

        llr = (np.clip(llr / self.llr_scale, -1.0, 1.0) * 127.0).astype(np.int8)
        
        target_ber = self.uncoded_ber[idx]

        hard = (llr < 0).astype(np.int8)
        current_errors = np.sum(hard != bits)

        target_errors = int(target_ber * N)
        delta = target_errors - current_errors
        
        if delta > 0:
            idx_sorted = np.argsort(np.abs(llr))

            candidates = idx_sorted[hard[idx_sorted] == bits[idx_sorted]]
            
            flip_idx = candidates[:delta]
            
            llr[flip_idx] *= -1
            
        elif delta < 0:
            idx_sorted = np.argsort(np.abs(llr))
            
            candidates = idx_sorted[hard[idx_sorted] != bits[idx_sorted]]
            
            fix_idx = candidates[:abs(delta)]
            
            llr[fix_idx] *= -1    
        
        hard = np.where(llr < 0, -1, 1).astype(np.int8)
        
        return llr