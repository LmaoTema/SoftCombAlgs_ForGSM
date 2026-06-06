import numpy as np
from .encoder import ConvolutionalEncoder
from .utils import MSC_PARAMS


class MSC1CRC:
    def __init__(self, parity_bits):
        self.parity_bits = parity_bits

    def encode(self, bits):
        return bits + [0] * self.parity_bits


class HeaderPuncturer:
    def __init__(self):
        self.forbidden = {26, 38, 50, 62, 74, 86, 98, 110, 113, 116}
        self.punct12 = [5, 8, 11]

    def process(self, bits):
        out = []
        for i, b in enumerate(bits):
            if i in self.forbidden or (i % 12 in self.punct12):
                continue
            out.append(b)
        return out


class DataPuncturer:
    def __init__(self, mode="P1"):
        self.mode = mode
        self.exceptions_p1 = {73, 136, 199, 262, 325, 388, 451, 514}
        self.exceptions_p2 = {78, 141, 204, 267, 330, 393, 456, 519}

    def process(self, bits):
        out = []
        for i, b in enumerate(bits):
            if self.mode == "P1":
                pos = i % 21
                if pos in [2, 5, 8, 10, 11, 14, 17, 20] and i not in self.exceptions_p1:
                    continue
            else:
                pos = i % 21
                if pos in [1, 4, 7, 9, 13, 15, 16, 19] and i not in self.exceptions_p2:
                    continue
            out.append(b)
        return out


class MSC1HeaderCoder:
    def __init__(self, params):
        self.crc = MSC1CRC(params["header_crc"])  # 8
        self.conv = ConvolutionalEncoder(
            G=[
                [1,1,1,1,0,1,1],
                [1,0,1,1,0,0,1],
                [1,1,0,1,1,0,1]
            ],
            K=7
        )
        self.puncturer = HeaderPuncturer()

    def process(self, header_bits: list):

        bits = self.crc.encode(header_bits)          # 31 + 8 = 39 бит
        
        coded = self.conv.process(bits)  
        coded_punctured = self.puncturer.process(coded)
        
        if len(coded_punctured) != 80:
            raise ValueError(f"Header puncturing error: {len(coded_punctured)} != 80")

        return coded_punctured


class MSC1DataCoder:
    def __init__(self, params, cps="P1"):
        self.crc = MSC1CRC(params["data_crc"])  # 12
        
        self.conv = ConvolutionalEncoder(
            G=[
                [1,1,1,1,0,1,1],
                [1,0,1,1,0,0,1],
                [1,1,0,1,1,0,1]
            ],
            K=7
        )
        self.puncturer = DataPuncturer(cps)

    def process(self, data_bits: list):
        bits = self.crc.encode(data_bits)        # 178 + 12 = 190
        bits = bits + [0] * 6                    # 190 + 6 = 196
        coded = self.conv.process(bits)          # 196 * 3 = 588
        coded = self.puncturer.process(coded)    # 372

        if len(coded) != 372:
            raise ValueError(f"Data puncturing error: {len(coded)} != 372")
        return coded


class MSC1Coding:
    def __init__(self, scheme="MCS1", cps="P1"):
        if scheme != "MCS1":
            raise ValueError("Only MCS-1 supported")
            
        params = MSC_PARAMS[scheme]
        self.header_coder = MSC1HeaderCoder(params)
        self.data_coder = MSC1DataCoder(params, cps)

    def process(self, bits: list):
        if len(bits) != 209:
            raise ValueError(f"Expected 209 bits, got {len(bits)}")
            
        header = bits[:31]
        data = bits[31:]

        h_encoded = self.header_coder.process(header)
        d_encoded = self.data_coder.process(data)

        full_encoded = h_encoded + d_encoded  # 80 + 372 = 452 bits

        return full_encoded + [0]*4