import numpy as np


class PDMRCCombiner:
    
    def combine(self, sector_soft_list):

        hards = []
        reliabilities = []
        llrs = []
        
        llrs = np.array(sector_soft_list)
        
        reliabilities = np.array(np.abs(llrs))
        hards = np.where(llrs < 0, -1, 1).astype(np.int8)
    
        metric = np.sum(reliabilities * hards, axis=0)

        combined_hard = np.sign(metric)
        combined_hard[combined_hard == 0] = 1 

        combined_reliability = np.abs(metric)

        combined_llr = combined_hard * combined_reliability
        
        return combined_llr