import numpy as np
from .viterbi_uni import ViterbiDecoder

class CS1Decoder:

    def __init__(self, vit_mode="vit_hard"):

        self.viterbi = ViterbiDecoder(
            constraint_length=5,
            polynomials=[0x13, 0x1B], mode=vit_mode
        )

    def process(self, bits):

        if not getattr(self, "is_working", True):
            return np.array(bits, dtype=int)

        coded = bits

        u = self.viterbi.decode(coded)  
        u = u[:184]  
        
        frame = np.array(u)

        return frame