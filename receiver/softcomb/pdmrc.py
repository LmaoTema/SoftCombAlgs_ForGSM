import numpy as np


class PDMRCCombiner:
    
    def combine(self, sector_soft_list):

        hards = []
        reliabilities = []

        hards = np.array([sector['hard'] for sector in sector_soft_list])
        reliabilities = np.array(np.abs([sector['llr'] for sector in sector_soft_list]))
    
        metric = np.sum(reliabilities * hards, axis=0)

        combined_hard = np.sign(metric)
        combined_hard[combined_hard == 0] = 1 

        combined_reliability = np.abs(metric)

        combined_llr = combined_hard * combined_reliability
        
        return combined_llr