import numpy as np


class PDMRCCombiner:
    def combine(self, sector_soft_list):
        if len(sector_soft_list) < 2:
            raise ValueError("PDMRC требует минимум 2 сектора")

        hards = []
        reliabilities = []

        hards = np.array([sector['hard'] for sector in sector_soft_list])
        reliabilities = np.array([sector['reliability'] for sector in sector_soft_list])
        
        assert hards.shape == reliabilities.shape, "Shape mismatch!"

        metric = np.sum(reliabilities * hards, axis=0)

        combined_hard = np.sign(metric)
        combined_hard[combined_hard == 0] = 1 

        combined_reliability = np.abs(metric)

        return {
            'hard': combined_hard,
            'reliability': combined_reliability
        }