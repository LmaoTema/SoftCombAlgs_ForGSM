import numpy as np
from transmitter.interleaver.msc_5 import MCS5Interleaver

class MCS5Deinterleaver:
    def extract_from_physical_bursts(self, physical_frame):
        
        if len(physical_frame) != 1872:
            raise ValueError(...)
        
        bursts_348 = []
        for i in range(4):
            burst = physical_frame[i*468 : (i+1)*468]
            e = np.zeros(348, dtype=int)
            
            e[0:174]   = burst[9:183]          # data1 + header1 + flags
            e[174:348] = burst[261:435]        # header2 + data2   (174 бит)
            
            bursts_348.append(e)
        
        return bursts_348

    def deinterleave_header(self, bursts_348):
        hi = np.zeros(136, dtype=int)
        
        for B in range(4):
            e = bursts_348[B]
            # header 1
            hi[34*B : 34*B + 18] = e[156:174]
            # header 2
            hi[34*B + 18 : 34*B + 34] = e[176:192]
        
        header_out = np.zeros(136, dtype=int)
        for k in range(136):
            j = 34 * (k % 4) + 2 * ((11 * k) % 17) + ((k % 8) // 4)
            header_out[k] = hi[j]
        
        return header_out

    def deinterleave_data(self, bursts_348):
            if not hasattr(MCS5Interleaver, '_data_mapping') or MCS5Interleaver._data_mapping is None:
                mapping = []
                k_prime = 0
                for k in range(1392):
                    B = k % 4
                    d = k % 464
                    j = 3 * (2 * ((25 * d) % 58) + ((d % 8) // 4) + 
                            2 * ((-1) ** B) * (d // 232)) + (k % 3)
                    if not (156 <= j <= 191):
                        if k_prime < 1248:
                            mapping.append((B, j))
                            k_prime += 1
            else:
                mapping = MCS5Interleaver._data_mapping
            
            data_out = np.zeros(1248, dtype=int)
            for idx, (B, j) in enumerate(mapping):
                data_out[idx] = bursts_348[B][j]
            
            return data_out

    def process(self, physical_frame):
        bursts_348 = self.extract_from_physical_bursts(physical_frame)
        header = self.deinterleave_header(bursts_348)
        data = self.deinterleave_data(bursts_348)
        frame = np.concatenate([header, data])
        
        return frame