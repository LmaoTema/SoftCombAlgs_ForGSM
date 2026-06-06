import numpy as np
from .viterbi_manager import ViterbiManager

class MSC5HeaderDepuncturer:
    def process(self, bits):

        if len(bits) != 136:
            raise ValueError(f"Header input must be 136 bits, got {len(bits)}")
        depunctured = bits[:135]  
        return depunctured


class MSC5DataDepuncturer:
    def __init__(self):

        self.exceptions_p1 = {47, 371, 695, 1019}
    def process(self, bits):
        if len(bits) != 1248:
            raise ValueError(f"Expected 1248 bits after deinterleaving, got {len(bits)}")
        out = []
        j = 0  

        for i in range(1404): 
            if (i >= 2 and (i - 2) % 9 == 0 and i <= 2 + 9*153) and i not in self.exceptions_p1:
                out.append(0)        
                continue
            if (i >= 1388 and (i - 1388) % 3 == 0 and i <= 1388 + 3*5) and i not in self.exceptions_p1:
                out.append(0)
                continue
            if j >= len(bits):
                raise ValueError("Depuncturer: недостаточно входных бит")
            out.append(bits[j])
            j += 1
        return out


class MSC5Decoder:
    def __init__(self, vit_mode="vit_soft", combining_method="PDMRC"):

        self.viterbi = ViterbiManager(
            constraint_length=7,
            polynomials=[0x7B, 0x59, 0x6D], 
            combining_method=combining_method, 
            mode=vit_mode
        )

        self.header_depunct = MSC5HeaderDepuncturer()
        self.data_depunct = MSC5DataDepuncturer()

    def process(self, input_data):

        if isinstance(input_data, np.ndarray):
            input_data = input_data.tolist()
                   
        if isinstance(input_data, (list, tuple)) and len(input_data) > 0:
            first_element = input_data[0]     
            
            if isinstance(first_element, (list, tuple, np.ndarray)):

                if len(input_data[0]) != 1384:
                    raise ValueError(f"Каждый сектор должен быть длиной 1384, получено {len(input_data[0])}")      

                header_sectors = []
                data_sectors = []
                
                for sector in input_data:
                    if len(sector) != 1384:
                        raise ValueError(f"Все сектора должны быть длиной 1384")
                    
                    h = self.header_depunct.process(sector[:136])
                    d = self.data_depunct.process(sector[136:])
                    header_sectors.append(h)
                    data_sectors.append(d)

                header_dec = self.viterbi.decode(header_sectors)
                data_dec = self.viterbi.decode(data_sectors)
                
            else:
                if len(input_data) != 1384:
                    raise ValueError(f"Ожидалось 1384 бита, получено {len(input_data)}")

                header_full = self.header_depunct.process(input_data[:136])
                data_full = self.data_depunct.process(input_data[136:])

                header_dec = self.viterbi.decode(header_full)
                data_dec = self.viterbi.decode(data_full)
                
        header_final = header_dec[:37]   
        data_final = data_dec[:450]     
        frame = header_final + data_final
        
        return frame
    
