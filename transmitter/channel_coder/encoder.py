import numpy as np
from core.block import Block

     
class ConvolutionalEncoder(Block):

    def __init__(self, G, K):
        
       # G - генеративные полиномы
       # K - кодовое ограничение
        
        self.G = G
        self.K = K
        self.state = [0]*(K-1)

    def encode_bit(self, bit):

        reg = [bit] + self.state

        out = []
        for g in self.G:

            val = 0
            for r, g_bit in zip(reg, g):
                if g_bit:
                    val ^= r # XOR операция 

            out.append(val)

        self.state = reg[:-1]

        return out

    def process(self, bits):

        output = []

        for b in bits:
            output.extend(self.encode_bit(b))

        return np.array(output)