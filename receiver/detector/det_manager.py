from core.block import Block
from .gmsk_det import GMSKDetector
from .psk_det import PSKDetector

class Detector(Block):

    def __init__(self, scheme, params, block_params, is_working=True):
        super().__init__(is_working)

        if scheme in ["TCHFS", "CS1", "MCS1"]:

            self.detector = GMSKDetector(params, block_params)

        elif scheme == "MCS5":

            self.detector = PSKDetector(params)

        else:

            raise ValueError("Unknown scheme")

    def _process(self, signal, h):

        return self.detector.process_detect(signal, h)
