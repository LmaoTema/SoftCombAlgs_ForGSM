
import numpy as np

from core.base import BasePipeline


class HalfPipeline(BasePipeline):

    def prepare_point(self, x_value, ber_ruler=None):

        self.point_index = ber_ruler.point_index
                
    def update_stats(self, ber_ruler, _, result, tx_bits):

        ber_ruler.update_frame(tx_bits=tx_bits, rx_bits=result["decoded_bits"], uncoded_ber=result["uncoded_ber"])
            
    def process(self, bits):

        coded_bits = self.encoder.process(bits.tolist())

        coded_bits = np.array(coded_bits)

        sector_soft_list = self.soft_llr_generator.get_soft_decisions(coded_bits, [self.point_index], num_sectors=2)[0]

        combined_llr = self.combiner.combine(sector_soft_list)

        decoded_bits = self.decoder.process(combined_llr)
        
        uncoded_ber = self.soft_llr_generator.get_uncoded_ber(self.point_index)

        return {
            "decoded_bits": decoded_bits,
            "uncoded_ber": uncoded_ber,
            "channel_output": None
        }
