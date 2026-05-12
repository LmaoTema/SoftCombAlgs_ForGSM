import numpy as np


class SCCombiner:
    def combine(self, sector_soft_list):
        
        hards = []
        reliabilities = []
        
        hards = np.array([s['hard'] for s in sector_soft_list])           
        reliabilities = np.array(np.abs([sector['llr'] for sector in sector_soft_list]))
        
        best_idx = np.argmax(np.abs(reliabilities), axis=0) 

        n_bits = hards.shape[1]
        idx = np.arange(n_bits)

        combined_hard = hards[best_idx, idx]
        combined_reliability = reliabilities[best_idx, idx]
        
        combined_llr = combined_hard * combined_reliability

        return combined_llr