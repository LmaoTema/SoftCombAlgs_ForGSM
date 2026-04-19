from receiver.softcomb.pdmrc import PDMRCCombiner
from receiver.softcomb.seleccomb import SCCombiner


class CombManager:
    def __init__(self, method,  is_working=False):
        
        super().__init__(is_working)
        
        if method == "PDMRC":
            self.combiner = PDMRCCombiner()

        elif method == "SC":
            self.combiner = SCCombiner()

        elif method == "ACS":
            self.combiner = None 

        else:
            raise ValueError(f"Unknown combining method: {method}")

    def combine(self, method, sector_soft_list):
        if self.method in ["PDMRC", "SC"]:
            return self.combiner.combine(sector_soft_list)

        elif method == "ACS":
            return sector_soft_list

    def is_acs(self):
        return self.method == "ACS"