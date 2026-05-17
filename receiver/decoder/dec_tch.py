import numpy as np
from .viterbi_manager import ViterbiManager

class TCHFSDecoder:

    def __init__(self, vit_mode="vit_soft", combining_method="PDMRC"):
        self.vit_mode = vit_mode
        self.combining_method = combining_method
        self.viterbi = ViterbiManager(
            constraint_length=5,
            polynomials=[0x13, 0x1B], combining_method=combining_method, mode=vit_mode
        )

    def process(self, bits):

        if not getattr(self, "is_working", True):
            return np.array(bits, dtype=int)


        if isinstance(bits, list):
            coded_parts = [llr[:378] for llr in bits]  
            class2 = bits[0][378:]                    
        else:
            coded_parts = bits[:378]
            class2 = bits[378:]

        u = self.viterbi.decode(coded_parts)  
        u = u[:189]  

        class1a_crc = u[:50]        # 53 - 3 CRC
        class1b = u[53:185]         # 132 bits
        
        if self.vit_mode == "vit_soft":
            class2 = (class2 <= 0).astype(np.uint8)

        frame = np.array(class1a_crc + class1b + list(class2))

        return frame