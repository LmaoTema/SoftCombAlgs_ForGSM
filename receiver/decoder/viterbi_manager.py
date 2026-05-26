import numpy as np
from .viterbi_uni import ViterbiDecoder
from .combing_viterbi import ViterbiDecoderACS   

class ViterbiManager:
    def __init__(self, constraint_length: int, polynomials: list, 
                 combining_method: str = "PDMRC", mode: str = "vit_soft"):
        
        self.combining_method = combining_method.upper()
        self.mode = mode
        
        self.single_viterbi = ViterbiDecoder(
            constraint_length=constraint_length,
            polynomials=polynomials,
            mode=mode
        )

        if self.combining_method == "ACS":
            self.multi_viterbi = ViterbiDecoderACS(
                constraint_length=constraint_length,
                polynomials=polynomials
            )

    def decode(self, input_data):

        if not isinstance(input_data, (list, tuple)):
            return self.single_viterbi.decode(input_data)

        # Если пришёл список списков
        if isinstance(input_data[0], (list, tuple, np.ndarray)):
            if self.combining_method == "ACS":
                return self.multi_viterbi.decode(input_data)
            else:
                from receiver.softcomb.comb_manager import CombManager
                combiner = CombManager(method=self.combining_method)
                combined_llr = combiner.combine(input_data)
                return self.single_viterbi.decode(combined_llr)

        else:
            return self.single_viterbi.decode(input_data)