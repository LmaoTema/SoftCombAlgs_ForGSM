from .encoder import ConvolutionalEncoder
import random

class CRC5350:
    def __init__(self):
        self.parity_bits = 3

    def encode(self, bits):
        parity = [random.randint(0, 1) for _ in range(self.parity_bits)]
        return bits + parity  # 50 bits + 3 parity bits


class TCHFSBlockCoder:
    def __init__(self):
        self.crc = CRC5350()

        G = [
            [1,0,0,1,1],  # G0
            [1,1,0,1,1]   # G1
        ]
        self.conv = ConvolutionalEncoder(G, 5)

    def process(self, bits):
        if not getattr(self, "is_working", True):
            return bits.copy()
        if len(bits) != 260:
            raise ValueError("TCH/FS frame must be 260 bits")

        class1a = bits[0:50]
        class1b = bits[50:182]  # 132 bits
        class2  = bits[182:260] # 78 bits

        
        class1a_crc = self.crc.encode(class1a)[50:]  

        u = class1a + list(class1a_crc) + class1b  
        tail_bits = [0]*4
        u += tail_bits 

        coded_class1 = self.conv.process(u)  

        # Итоговый frame
        frame = list(coded_class1) + list(class2)  # 378 + 78 = 456 
        return frame