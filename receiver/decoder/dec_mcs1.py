import numpy as np
from .viterbi_manager import ViterbiManager
from transmitter.channel_coder.utils import MSC_PARAMS


class HeaderDepuncturer:
    def __init__(self):
        self.forbidden = {26, 38, 50, 62, 74, 86, 98, 110, 113, 116}
        self.punct12 = [5, 8, 11]

    def process(self, bits):
        out = []
        j = 0
        for i in range(117):
            if i in self.forbidden or (i % 12 in self.punct12):
                out.append(0)
            else:
                if j >= len(bits):
                    raise ValueError("Header depuncturing: недостаточно бит")
                out.append(bits[j])
                j += 1
        return out


class DataDepuncturer:
    def __init__(self, mode="P1"):
        self.mode = mode
        self.exceptions_p1 = {73, 136, 199, 262, 325, 388, 451, 514}
        self.exceptions_p2 = {78, 141, 204, 267, 330, 393, 456, 519}

    def process(self, bits):
        out = []
        j = 0
        for i in range(588):
            if self.mode == "P1":
                pos = i % 21
                if pos in [2, 5, 8, 10, 11, 14, 17, 20] and i not in self.exceptions_p1:
                    out.append(0)
                else:
                    if j >= len(bits):
                        raise ValueError("Data depuncturing: недостаточно бит")
                    out.append(bits[j])
                    j += 1
            else:
                pos = i % 21
                if pos in [1, 4, 7, 9, 13, 15, 16, 19] and i not in self.exceptions_p2:
                    out.append(0)
                else:
                    if j >= len(bits):
                        raise ValueError("Data depuncturing: недостаточно бит")
                    out.append(bits[j])
                    j += 1
        return out


class MSC1Decoder:
    def __init__(self, cps="P1", vit_mode="vit_soft", combining_method="PDMRC"):
        params = MSC_PARAMS["MCS1"]

        self.header_crc_len = params["header_crc"]
        self.data_crc_len = params["data_crc"]
        
        self.viterbi = ViterbiManager(
            constraint_length=7,
            polynomials=[0x7B, 0x59, 0x6D], 
            combining_method=combining_method, 
            mode=vit_mode
        )

        self.header_depunct = HeaderDepuncturer()
        self.data_depunct = DataDepuncturer(cps)

    def process(self, input_data):

        if isinstance(input_data, np.ndarray):
            input_data = input_data.tolist()


        if isinstance(input_data, (list, tuple)) and len(input_data) > 0:
            first_element = input_data[0]
            

            if isinstance(first_element, (list, tuple, np.ndarray)):

                if len(input_data[0]) != 456:
                    raise ValueError(f"Каждый сектор должен быть длиной 452, получено {len(input_data[0])}")

                header_sectors = []
                data_sectors = []

                for sector in input_data:
                    if len(sector) != 456:
                        raise ValueError(f"Все сектора должны быть длиной 452")
                    
                    h = self.header_depunct.process(sector[:80])
                    d = self.data_depunct.process(sector[80:452])
                    header_sectors.append(h)
                    data_sectors.append(d)

                header_dec = self.viterbi.decode(header_sectors)
                data_dec = self.viterbi.decode(data_sectors)

            else:
                if len(input_data) != 456:
                    raise ValueError(f"Ожидалось 452 бита, получено {len(input_data)}")

                header_full = self.header_depunct.process(input_data[:80])
                data_full = self.data_depunct.process(input_data[80:452])

                header_dec = self.viterbi.decode(header_full)
                data_dec = self.viterbi.decode(data_full)

        else:
            raise ValueError(f"Неверный формат входных данных: {type(input_data)}")


        header_final = header_dec[:31]
        data_final = data_dec[:178]

        return header_final + data_final