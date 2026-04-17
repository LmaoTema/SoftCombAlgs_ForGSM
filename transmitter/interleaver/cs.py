import numpy as np

class CS1Interleaver:

    def __init__(self):

        self.tail_bits = np.zeros(3, dtype=int)
        self.guard_bits = np.zeros(8, dtype=int)
        self.st_flag = np.zeros(1, dtype=int)        

        self.training_seq = np.array([
            0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 0, 1, 0, 1, 1, 0, 1, 1, 1
        ], dtype=int)  

    def process(self, bits):
        if len(bits) != 456:
            raise ValueError("CS1Interleaver: ожидается ровно 456 бит")

        bursts_data = np.zeros((4, 114), dtype=int)   

        for k in range(456):
            B = k % 4                                
            j = 2 * ((49 * k) % 57) + ((k % 8) // 4)  
            bursts_data[B, j] = bits[k]
        bursts = []

        for b in range(4):
            data = bursts_data[b]          

            left_data = data[:57]          
            right_data = data[57:]         

            burst = np.concatenate([
                self.tail_bits,      # 3
                left_data,           # 57
                self.st_flag,        # 1
                self.training_seq,   # 26
                right_data,          # 57
                self.st_flag,        # 1
                self.tail_bits,      # 3
                self.guard_bits      # 8
            ])  

            bursts.append(burst)
            
        return np.concatenate(bursts)