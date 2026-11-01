from core.block import Block
from transmitter.interleaver.tch import TCHInterleaver
from transmitter.interleaver.cs import CS1Interleaver
from transmitter.interleaver.msc_1 import MCS1Interleaver
from transmitter.interleaver.msc_5 import MCS5Interleaver


class Interleaver(Block):

    def __init__(self, scheme, is_working=False, CCI_enabled=False):

        super().__init__(is_working)

        if scheme == "TCHFS":
            self.interleaver = TCHInterleaver(CCI_enabled)

        elif scheme == "CS1":
            self.interleaver = CS1Interleaver(CCI_enabled)

        elif scheme == "MCS1":
            self.interleaver = MCS1Interleaver(CCI_enabled)

        elif scheme == "MCS5":
            self.interleaver = MCS5Interleaver(CCI_enabled)

        else:
            raise ValueError("Unknown scheme")

    def _process(self, bits):

        return self.interleaver.process(bits)