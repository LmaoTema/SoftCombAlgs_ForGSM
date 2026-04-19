import numpy as np

class SoftGenerator:
    def __init__(self, num_sectors, config):
        self.num_sectors = num_sectors
        self.config = config
        
        
    def get_soft_decisions(self, tx_bits, snr_db):
        soft_list = []
        for snr_db in snr_db:
            soft_sector = self._simulate_sova_soft_for_sector(tx_bits, snr_db)
            soft_list.append(soft_sector)

        return soft_list

    def _simulate_sova_soft_for_sector(self, tx_bits, snr_db):

        # пока нет - ждём распредление 
        return {
            'hard': hard,           
            'reliability': reliability 
        }