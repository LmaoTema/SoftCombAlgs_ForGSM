import numpy as np
from .viterbi_uni import ViterbiDecoder

class CRC5350Decoder:

    def __init__(self):
        self.crc_errors = 0 
        self.poly = [1,0,1,1]

    def check(self, bits):

        reg = bits.copy()

        for i in range(50):

            if reg[i] == 1:
                for j in range(len(self.poly)):
                    reg[i+j] ^= self.poly[j]

        return reg[-3:] == [0,0,0]

    def decode(self, bits):

        data = bits[:50]

        if not self.check(bits):
            self.crc_errors += 1

        return data
    
class TCHFSDecoder:

    def __init__(self):

        self.viterbi = ViterbiDecoder(
            constraint_length=5,
            polynomials=[0x13, 0x1B]
        )

        self.crc = CRC5350Decoder()

    def process(self, bits):

        if not getattr(self, "is_working", True):
            return np.array(bits, dtype=int)

        coded_class1 = bits[:378]  # 378 бит 
        class2 = bits[378:]        # 78 бит class2

        u = self.viterbi.decode(coded_class1)  
        u = u[:189]  

        class1a_crc = u[:53]        # 50 + 3 CRC
        class1b = u[53:185]         # 132 bits

        class1a = self.crc.decode(class1a_crc)
        
        frame = np.array(class1a + class1b + list(class2))

        return frame