import numpy as np
from core.block import Block
from .zero_force import ZFEqualizer
from .dfe import DFEEqualizer

class Equalizer(Block):

    def __init__(self, equalizer_params, modulation_params, is_working=True):

        type_demod = modulation_params.get("type_demod", "diff_phase")
        if (type_demod in ["vit_hard", "vit_soft"]):
            is_working = False

        super().__init__(is_working)

        eq_type = equalizer_params.get("equalizer_type", "ZF")
        channel_model = equalizer_params.get("channel_model", "awgn")

        if eq_type == "ZF":
            self.equalizer = ZFEqualizer(modulation_params, channel_model)
        elif eq_type == "DFE":
            self.equalizer = DFEEqualizer(modulation_params, channel_model)
    
    def _process(self, match_signal, h):

        return self.equalizer.process_eq(match_signal,)