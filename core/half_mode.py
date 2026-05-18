
import numpy as np

from core.base import BasePipeline


class HalfPipeline(BasePipeline):

    def prepare_point(self, x_value, ber_ruler=None):

        self.point_index = ber_ruler.point_index
                
    def update_stats(self, ber_ruler, _, result, tx_bits):

        ber_ruler.update_frame(tx_bits=tx_bits, rx_bits=result["decoded_bits"], uncoded_ber=result["uncoded_ber"])
            
    def process(self, bits):

        coded_bits = self.encoder.process(bits.tolist())

        interleaved_bits = np.array(self.interleaver.process(coded_bits))
        interleaved_bits = interleaved_bits.reshape(-1, 156)[:, :148].reshape(-1)
        
        sector_soft_list = self.soft_llr_generator.get_soft_decisions(interleaved_bits, [self.point_index], num_sectors=2)[0]
        
        sector_deinterleaved = []
        for llr in sector_soft_list:
            deintl_llr = self.deinterleaver.process(llr)     
            sector_deinterleaved.append(deintl_llr)
            
        decoded_bits = self.decoder.process(sector_deinterleaved)
        
        uncoded_ber = self.soft_llr_generator.get_uncoded_ber(self.point_index)

        return {
            "decoded_bits": decoded_bits,
            "uncoded_ber": uncoded_ber,
            "channel_output": None
        }
