from core.block import Block
from receiver.decoder.dec_tch import TCHFSDecoder
from receiver.decoder.dec_cs1 import CS1Decoder
from receiver.decoder.dec_mcs1 import MSC1Decoder

class ChannelDecoder(Block):
    def __init__(self, scheme, vit_mode="vit_hard", is_working=True):
        super().__init__(is_working)
        
        if scheme == "TCHFS":
            self.decoder = TCHFSDecoder(vit_mode=vit_mode)  
        elif scheme == "CS1":
            self.decoder = CS1Decoder(vit_mode=vit_mode)
        elif scheme == "MCS1":
            self.decoder = MSC1Decoder()
        elif scheme == "MCS1":
            self.decoder = MSC5Decoder()

    def _process(self, bits):
        return self.decoder.process(bits)