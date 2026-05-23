
import numpy as np

from core.base import BasePipeline


class HalfPipeline(BasePipeline):

    def prepare_point(self, x_value, ber_ruler=None):

        self.point_index = ber_ruler.point_index
                
    def update_stats(self, ber_ruler, ber_ruler_uncoded, result, tx_bits):

        ber_ruler.update_frame(tx_bits=tx_bits, rx_bits=result["decoded_bits"], uncoded_ber=result["uncoded_ber"])
        
        ber_ruler_uncoded.update_frame(
            tx_bits=result["tx_bits"],
            rx_bits=result["rx_bits"],
            )

            
    def process(self, bits):

        coded_bits = self.encoder.process(bits.tolist())

        interleaved_bits = np.array(self.interleaver.process(coded_bits))
        interleaved_bits = interleaved_bits.reshape(-1, 156)[:, :148].reshape(-1)
        
        sector_soft_list = self.soft_llr_generator.get_soft_decisions(interleaved_bits, [self.point_index], num_sectors=2)[0]
        
        # sector_deinterleaved = self.deinterleaver.process(sector_soft_list)
        sector_deinterleaved = []
        for llr in sector_soft_list:
            deintl_llr = self.deinterleaver.process(llr)     
            sector_deinterleaved.append(deintl_llr)
        
        decoded_bits = self.decoder.process(sector_deinterleaved)
        
        for_hard_llr = sector_soft_list[0]
        hard_list_dec = (for_hard_llr < 0).astype(np.int8)
        uncoded_ber = self.soft_llr_generator.get_uncoded_ber(self.point_index)

        return {
            "decoded_bits": decoded_bits,
            "uncoded_ber": uncoded_ber, 
            "tx_bits": interleaved_bits,
            "rx_bits": hard_list_dec,
            "channel_output": None
        }
