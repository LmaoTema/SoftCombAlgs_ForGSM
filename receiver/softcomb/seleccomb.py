import numpy as np


class SCCombiner:
    def combine(self, sector_soft_list):
        
        hards = []
        reliabilities = []
        
        llrs = []
        
        llrs = np.array(sector_soft_list)
        
        reliabilities = np.array(np.abs(llrs))
        hards = np.where(llrs < 0, -1, 1).astype(np.int8)
        
        best_idx = np.argmax(np.abs(reliabilities), axis=0) 

        n_bits = hards.shape[1]
        idx = np.arange(n_bits)

        combined_hard = hards[best_idx, idx]
        combined_reliability = reliabilities[best_idx, idx]
        
        combined_llr = combined_hard * combined_reliability

        return combined_llr