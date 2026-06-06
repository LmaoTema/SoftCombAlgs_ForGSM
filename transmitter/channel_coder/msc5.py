import numpy as np
from .encoder import ConvolutionalEncoder
from .utils import MSC_PARAMS

class MSC5CRC:
    def __init__(self, parity_bits):
        self.parity_bits = parity_bits

    def encode(self, bits):
        parity = [0] * self.parity_bits 
        return bits + parity

class MSC5HeaderCoder:
    def __init__(self, params):
        self.crc = MSC5CRC(params["header_crc"])
        G = [
            [1,1,1,1,0,1,1],
            [1,0,1,1,0,0,1],
            [1,1,0,1,1,0,1]
        ]
        self.conv = ConvolutionalEncoder(G, 7)
        self.use_puncture = params["header_puncture"]

    def process(self, bits):
       
        bits = self.crc.encode(bits)            # 37 + 8 = 45  
        coded = self.conv.process(bits)         # 45 * 3 = 135
        coded = np.append(coded, coded[-1])     # 135 + 1 = 136 
        return coded

class MCS5DataPuncturer:
    def __init__(self):
        self.exceptions_p1 = {47, 371, 695, 1019}
    def process(self, bits):
        out = []
        for i, b in enumerate(bits):
            if (i >= 2 and (i - 2) % 9 == 0 and i <= 2 + 9*153) and i not in self.exceptions_p1:
                continue
        
            if (i >= 1388 and (i - 1388) % 3 == 0 and i <= 1388 + 3*5) and i not in self.exceptions_p1:
                continue

            out.append(b)
        return out
    
class MSC5DataCoder:
    def __init__(self, params):
        self.crc = MSC5CRC(params["data_crc"])
        G = [
            [1,1,1,1,0,1,1],
            [1,0,1,1,0,0,1],
            [1,1,0,1,1,0,1]
        ]
        self.conv = ConvolutionalEncoder(G, 7)
        self.punct = MCS5DataPuncturer()

    def process(self, bits):
        bits = self.crc.encode(bits)
        bits = bits + [0] * 6 
        coded = self.conv.process(bits)
        coded = self.punct.process(coded)
        return coded

class MSC5Coding:
    def __init__(self, scheme):
        if scheme != "MCS5":
            raise ValueError("MSC-5 coder only")
        params = MSC_PARAMS[scheme]
        self.header = MSC5HeaderCoder(params)
        self.data = MSC5DataCoder(params)
        self.scheme = scheme

    def process(self, bits):
        header = bits[:37]
        data = bits[37:]

        h = self.header.process(header)     # 136 
        d = self.data.process(data)         # 1248 
        frame = np.concatenate([h, d])
        return frame