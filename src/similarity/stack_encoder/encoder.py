import torch
import torch.nn as nn
from torch import Tensor

from src.similarity.methods.neural.siam.encoders import EncoderModel
from src.similarity.preprocess.seq_coder import SeqCoder


class LSTMEncoder(EncoderModel):
    # https://discuss.pytorch.org/t/bidirectional-3-layer-lstm-hidden-output/41336/4
    def __init__(self, coder: SeqCoder, dim: int = 50, hid_dim: int = 200,
                 bidir: bool = False, **kvargs):
        super(LSTMEncoder, self).__init__(f"hdim={hid_dim}",
                                          coder, dim, out_dim=hid_dim, **kvargs)
        self.bidir = bidir
        self.hidden_dim = hid_dim
        self.lstm = nn.LSTM(dim, self.hidden_dim, batch_first=True, bidirectional=bidir)

    def forward(self, stack_id: int) -> Tensor:
        emb_f = self.word_embeddings(self.to_inds(stack_id))
        emb_f = emb_f.view(1, emb_f.shape[0], -1)  # [1, L, dim]
        output, (hn, cn) = self.lstm(emb_f)

        if not self.bidir:
            return hn[-1]

        h_1, h_2 = hn[0], hn[1]

        return torch.cat((h_1, h_2), 1)
