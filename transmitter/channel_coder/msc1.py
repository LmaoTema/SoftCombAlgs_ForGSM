import numpy as np
from .encoder import ConvolutionalEncoder
from .utils import MSC_PARAMS, prepend_last_bits


class MSC1CRC:
    def __init__(self, parity_bits):
        self.parity_bits = parity_bits

    def encode(self, bits):
        # zaglushka
        parity = [0]*self.parity_bits
        return bits + parity

class HeaderPuncturer:
    def __init__(self):
        self.forbidden = {26,38,50,62,74,86,98,110,113,116}
        self.punct12 = [5,8,11]

    def process(self, bits):
        out = []
        for i,b in enumerate(bits):
            if i in self.forbidden or (i%12 in self.punct12):
                continue
            out.append(b)
        return out

class DataPuncturer:
    def __init__(self, mode="P1"):
        self.mode = mode
        self.exceptions = {73,136,199,262,325,388,451,514}
        self.exceptions2 = {78,141,204,267,330,393,456,519}

    def process(self, bits):
        out = []
        for i,b in enumerate(bits):
            if self.mode == "P1":
                pos = i % 21
                if pos in [2,5,8,10,11,14,17,20] and i not in self.exceptions:
                    continue
            else:
                pos = i % 21
                if pos in [1,4,7,9,13,15,16,19] and i not in self.exceptions2:
                    continue
            out.append(b)
        return out


class MSC1HeaderCoder:
    def __init__(self, params):
        self.crc = MSC1CRC(params["header_crc"])
        G = [
            [1,1,1,1,0,1,1],
            [1,0,1,1,0,0,1],
            [1,1,0,1,1,0,1]
        ]
        self.conv = ConvolutionalEncoder(G,7)
        self.use_puncture = params["header_puncture"]

    def process(self, bits):
        bits = self.crc.encode(bits)
        coded = self.conv.process(bits)
        if self.use_puncture:
            coded = HeaderPuncturer().process(coded)
        return coded


class MSC1DataCoder:
    def __init__(self, params, cps="P1"):
        self.crc = MSC1CRC(params["data_crc"])
        G = [
            [1,1,1,1,0,1,1],
            [1,0,1,1,0,0,1],
            [1,1,0,1,1,0,1]
        ]
        self.conv = ConvolutionalEncoder(G,7)
        self.punct = DataPuncturer(cps)

    def process(self, bits):
        bits = self.crc.encode(bits)
        bits = bits + [0]*6
        self.beforecoder = bits[:]
        coded = self.conv.process(bits)
        coded = self.punct.process(coded)
        return coded


class MSC1Coding:
    def __init__(self, scheme, cps="P1"):
        if scheme != "MCS1":
            raise ValueError("MSC-1 coder only")
        params = MSC_PARAMS[scheme]
        self.header_bits = params["header_bits"]
        self.data_bits = params["data_bits"]
        self.header = MSC1HeaderCoder(params)
        self.data = MSC1DataCoder(params, cps)
        self.scheme = scheme

    def process(self, bits):
        header = bits[0:31]
        data = bits[31:209]
        
        h = self.header.process(header)
        d = self.data.process(data)
        
        coded = h + d
        coded = coded + [0]*4
        
        self.header_raw = header
        self.data_raw = data

        self.header_encoded = h
        self.data_encoded = d

        self.full_encoded = coded
        
        return coded 