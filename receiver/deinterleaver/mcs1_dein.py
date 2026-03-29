import numpy as np

class MCS1Deinterleaver:
    def __init__(self):
        self.bursts = 4       
        self.data_len = 57    
    def deinterleave_57(self, subblock):
        out = np.zeros(57, dtype=int)
        for k in range(57):
            j = (49 * k) % 57
            out[k] = subblock[j]  # или правильный порядок
        return out

    def process(self, bits):

        if len(bits) == 624:
            bursts = bits.reshape(self.bursts, 156)
            bursts = bursts[:, :148]  
        elif len(bits) == 592:
            bursts = bits.reshape(self.bursts, 148)  
        else:
            raise ValueError("Input bits must be 624 or 592 bits")


        subblocks = [np.zeros(57, dtype=int) for _ in range(8)]

        for b in range(self.bursts):

            first_data = bursts[b, :self.data_len]
            second_data = bursts[b, self.data_len:2*self.data_len]

            subblocks[b] = self.deinterleave_57(first_data)
            subblocks[b + 4] = self.deinterleave_57(second_data)

        return np.concatenate(subblocks)  