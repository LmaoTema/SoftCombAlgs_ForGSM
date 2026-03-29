import numpy as np

class SpeechDeinterleaver:
    def __init__(self):
        pass

    def deinterleave_57(self, data):

        out = np.zeros(57, dtype=int)

        for j in range(57):
            k = (7 * j) % 57
            out[k] = data[j]

        return out

    def extract_data(self, burst):

        first = burst[3:60]
        second = burst[87:144]

        return first, second

    def process(self, bits):

        bits = np.array(bits)

        if len(bits) == 624:
             bursts = bits.reshape(4, 156)
             bursts = bursts[:, :148]
        else:
            bursts = bits.reshape(4, 148)

        subblocks = [None] * 8

        for b in range(4):
            first, second = self.extract_data(bursts[b])

            subblocks[b] = self.deinterleave_57(first)
            subblocks[b + 4] = self.deinterleave_57(second)

        bits = np.concatenate(subblocks)

        return bits