import numpy as np
from .viterbi_uni import ViterbiDecoder
from transmitter.channel_coder.utils import MSC_PARAMS


class MSC1Decoder:

    def __init__(self):

        params = MSC_PARAMS["MCS1"]

        self.header_bits = params["header_bits"]
        self.data_bits = params["data_bits"]

        self.header_crc = params["header_crc"]
        self.data_crc = params["data_crc"]

        self.viterbi = ViterbiDecoder(
            constraint_length=7,
            polynomials = [0x7B, 0x59, 0x6D]
        )

    def _depuncture_header(self, bits):
        forbidden = {26,38,50,62,74,86,98,110,113,116}
        punct12 = [5,8,11]

        out = []
        j = 0
        for i in range(117):  # 117 = 39*3, точное количество кодированных бит
            if i in forbidden or (i % 12 in punct12):
                out.append(None)  # erasure
            else:
                if j >= len(bits):
                    raise ValueError("Header depuncturing: not enough input bits")
                out.append(bits[j])
                j += 1
        return out

    def _depuncture_data(self, bits):
        exceptions = {73,136,199,262,325,388,451,514}
        out = []
        j = 0
        for i in range(588):  
            pos = i % 21
            if pos in [2,5,8,10,11,14,17,20] and i not in exceptions:
                out.append(None) 
            else:
                if j >= len(bits):
                    raise ValueError("Data depuncturing: not enough input bits")
                out.append(bits[j])
                j += 1
        return out

    def process(self, bits):

        if not getattr(self, "is_working", True):
            return np.array(bits, dtype=int)

        bits = bits[:-4]
        header_coded = bits[:80]
        data_coded = bits[80:452] 
        
        header_full = self._depuncture_header(header_coded)
        data_full = self._depuncture_data(data_coded)
        
        h_dec = self.viterbi.decode(header_full)
        d_dec = self.viterbi.decode(data_full)

        d_dec = d_dec[:-6]
        
        h_dec = h_dec[:-self.header_crc]
        d_dec = d_dec[:-self.data_crc]

        frame = np.array(h_dec + d_dec, dtype=int)

        return frame