import numpy as np
from .encoder import ConvolutionalEncoder
from core.block import Block
import random

class FIRECode:
    def __init__(self):
        self.parity_bits = 40

    def encode(self, bits):
        if len(bits) != 184:
            raise ValueError("CS-1 block must be 184 bits")
        parity = [random.randint(0, 1) for _ in range(self.parity_bits)]
        return bits + parity  


class CS1BlockCoder:
    def __init__(self):
        self.fire = FIRECode()

        G = [
            [1,0,0,1,1], 
            [1,1,0,1,1]   
        ]
        
        self.conv = ConvolutionalEncoder(G,5)

    def process(self, bits):
        if len(bits) != 184:
            raise ValueError("CS-1 frame must be 184 bits")

        block224 = self.fire.encode(bits)  

        tail = [0,0,0,0]
        u = block224 + tail  # 228 bits

        coded = self.conv.process(u)
    
        return coded