import numpy as np


class SpeechDeinterleaver:
    def __init__(self):
        pass

    def extract_data(self, burst):
        left = burst[3:60]
        right = burst[87:144]
        return np.concatenate([left, right])  

    def process(self, bits):

        bits = np.array(bits)
        
        bursts = bits.reshape(8, 148)
        bursts_data = np.zeros((8, 114), dtype=int)

        for b in range(8):
            bursts_data[b] = self.extract_data(bursts[b])

        output = np.zeros(456, dtype=int)

        for k in range(456):
            B = k % 8
            j = 2 * ((49 * k) % 57) + ((k % 8) // 4)

            output[k] = bursts_data[B, j]

        return output