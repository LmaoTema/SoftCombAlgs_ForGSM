import numpy as np

class MCS5Interleaver:
    def __init__(self):

        self.training_seq = np.array([1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,
        1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,0,0,1], dtype=int)

        self.data_mapping = None
        self.tail_bits = np.ones(9, dtype=int)      # 9 
        self.guard_bits = np.zeros(24, dtype=int)   # 24 

    @staticmethod
    def header_interleave(header_bits):
        if len(header_bits) != 136:
            raise ValueError(f"Header must have 136 bits, got {len(header_bits)}")
        
        hi = np.zeros(136, dtype=int)
        for k in range(136):
            j = 34 * (k % 4) + 2 * ((11 * k) % 17) + ((k % 8) // 4)
            hi[j] = header_bits[k]
        return hi

    def data_interleave(self, dc_bits):
        if len(dc_bits) != 1248:
            raise ValueError(f"Data must have 1248 bits, got {len(dc_bits)}")
        
        bursts = [np.zeros(348, dtype=int) for _ in range(4)]
        mapping = []
        
        k_prime = 0
        for k in range(1392):
            B = k % 4
            d = k % 464
            sign = 1 if B % 2 == 0 else -1
            j = 3 * (2 * ((25 * d) % 58) + ((d % 8) // 4) + 
                     2 * sign * (d // 232)) + (k % 3)
            
            if not (156 <= j <= 191):
                if k_prime < 1248:
                    bursts[B][j] = dc_bits[k_prime]
                    mapping.append((B, j))
                    k_prime += 1
        
        self._data_mapping = mapping
        return bursts  


    def map_to_bursts(self, hi, di_bursts):
        bursts_348 = []
        q = np.zeros(8, dtype=int)  # stealing flags

        for B in range(4):
            e = np.zeros(348, dtype=int)
    
            e[:] = di_bursts[B][:]        
        
            e[156:174] = hi[34*B : 34*B + 18]
            e[174:176] = q[2*B : 2*B + 2]
            e[176:192] = hi[34*B + 18 : 34*B + 34]
            
            bursts_348.append(e)
        
        return bursts_348

    def build_physical_bursts(self, bursts_348):
        physical_bursts = []

        for e in bursts_348:
            burst = np.zeros(468, dtype=int)

            # tb 
            burst[0:9] = self.tail_bits

            # data
            burst[9:183] = e[0:174]

            # ts
            burst[183:261] = self.training_seq

            # data
            burst[261:435] = e[174:348]

            # tb
            burst[435:444] = self.tail_bits

            # gp 
            burst[444:468] = self.guard_bits

            physical_bursts.append(burst)

        return physical_bursts

    def process(self, bits):
        header = np.array(bits[0:136], dtype=int)
        data = np.array(bits[136:], dtype=int)
        hi = self.header_interleave(header)
        di_bursts = self.data_interleave(data)

        bursts_348 = self.map_to_bursts(hi, di_bursts)
        physical_bursts = self.build_physical_bursts(bursts_348)

        frame_total = np.concatenate(physical_bursts)
        return frame_total