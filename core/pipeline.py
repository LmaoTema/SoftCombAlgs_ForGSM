class ProcessingMode:
    NONE = "none"
    HALF = "half"
    FULL = "full"
    
    def sweep(self):
        if self.mode == ProcessingMode.HALF:
            return self.soft_llr_generator.rssi_db

        if self.mode in [ProcessingMode.FULL, ProcessingMode.NONE]:
            return None  # BERRuler сам управляет sweep

        return None
