from ctcdecode import CTCBeamDecoder
import torch.nn.functional as F
import torch
import numpy as np
import editdistance


class Decoder:

    def __init__(self, labels, lm_path=None, alpha=1, beta=1.5, cutoff_top_n=40, cutoff_prob=0.99, beam_width=200, num_processes=24, blank_id=0):
        self.vocab_list = ['_'] + labels # NOTE: blank symbol
        self._decoder = CTCBeamDecoder(['_'] + labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width, num_processes, blank_id)


    def convert_to_string(self, tokens, seq_len=None):
        if not seq_len:
            seq_len = tokens.size(0)
        out = []
        for i in range(seq_len):
            if len(out) == 0:
                if tokens[i] != 0:
                    out.append(tokens[i])
            else:
                if tokens[i] != 0 : # and tokens[i] != tokens[i - 1]:
                    out.append(tokens[i])
        return ''.join(self.vocab_list[i] for i in out)

    def decode_beam(self, tlogits, seq_lens):
        decoded = []
        #tlogits = logits.transpose(0, 1)
        beam_result, beam_scores, timesteps, out_seq_len = self._decoder.decode(tlogits, seq_lens)
        for i in range(tlogits.size(0)):
            output_str = ''.join(map(lambda x: self.vocab_list[x], beam_result[i][0][:out_seq_len[i][0]]))
            decoded.append(output_str)
        return decoded

    def decode_greedy(self, logits, seq_lens):
        decoded = []
        tlogits = logits.transpose(0, 1)
        print(tlogits.size())
        _, tokens = torch.max(tlogits, 2)
        for i in range(tlogits.size(0)):
            output_str = self.convert_to_string(tokens[i], seq_lens[i])
            decoded.append(output_str)
        return decoded

    def get_mean(self, decoded, gt, individual_length, func):
        total_norm  = 0.0
        length      = len(decoded)
        for i in range(0, length):
            val         = float(func(decoded[i], gt[i]))
            total_norm += val / individual_length
        return total_norm / length

    def cer_batch(self, decoded, gt):
        assert len(decoded) == len(gt), 'batch size mismatch'
        mean_indiv_len = np.mean([len(s) for s in gt])

        return self.get_mean(decoded, gt, mean_indiv_len, editdistance.eval)
