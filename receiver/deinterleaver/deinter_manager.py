from core.block import Block
from receiver.deinterleaver.tch_dein import SpeechDeinterleaver
from receiver.deinterleaver.cs_dein import CS1Deinterleaver
from receiver.deinterleaver.mcs1_dein import MCS1Deinterleaver

class Deinterleaver(Block):

    def __init__(self, scheme, is_working=False):

        super().__init__(is_working)

        if scheme == "TCHFS":
            self.deinterleaver = SpeechDeinterleaver()

        elif scheme == "CS1":
            self.deinterleaver = CS1Deinterleaver()

        elif scheme == "MCS1":
            self.deinterleaver = MCS1Deinterleaver()

        elif scheme == "MCS5":
            self.deinterleaver = MCS5Deinterleaver()

        else:
            raise ValueError("Unknown scheme")

    def _process(self, bits):

        return self.deinterleaver.process(bits)