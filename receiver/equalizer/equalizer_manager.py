import numpy as np
from core.block import Block
from .zero_force import ZFEqualizer
from .dfe import DFEEqualizer

class Equalizer(Block):

    def __init__(self, equalizer_params, modulation_params, is_working=True):

        eq_type = equalizer_params.get("equalizer_type", "ZF")
        channel_model = equalizer_params.get("channel_model", "awgn")

        type_demod = modulation_params.get("type_demod", "diff_phase")

        if eq_type != "DFE" and type_demod in ["vit_hard", "vit_soft"]:
            is_working = False

        super().__init__(is_working)

        self.eq_type = eq_type
        self.provides_soft = (eq_type == "DFE")

        if eq_type == "ZF":
            self.equalizer = ZFEqualizer(modulation_params, channel_model)
        elif eq_type == "DFE":
            self.equalizer = DFEEqualizer(modulation_params, channel_model)

    def _process(self, match_signal, h):
        out = self.equalizer.process_eq(match_signal, h)

        if isinstance(out, tuple):
            eq_signal, llr = out
        else:
            eq_signal, llr = out, None

        return eq_signal, llr
