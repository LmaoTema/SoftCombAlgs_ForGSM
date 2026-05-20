from receiver.softcomb.pdmrc import PDMRCCombiner
from receiver.softcomb.seleccomb import SCCombiner


class CombManager:
    def __init__(self, method):
        self.method = method

        if method == "PDMRC":
            self.combiner = PDMRCCombiner()

        elif method == "SC":
            self.combiner = SCCombiner()
        else:
            raise ValueError(f"Unknown combining method: {method}")

    def combine(self, sector_soft_list):
            if self.method in ["PDMRC", "SC", "ACS"]:
                return self.combiner.combine(sector_soft_list) 
            else:
                raise ValueError(f"Unsupported method: {self.method}")