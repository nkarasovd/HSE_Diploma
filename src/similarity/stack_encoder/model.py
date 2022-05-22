from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch import cosine_similarity

from src.similarity.methods.neural import device
from src.similarity.stack_encoder.encoder import LSTMEncoder


class StackEncoder(nn.Module):
    def __init__(self, coder, hidden_dim: int = 100, embed_dim: int = 128):
        super(StackEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.encoder = LSTMEncoder(coder, dim=50, hid_dim=hidden_dim).to(device)

    def forward(self, stack_id: int) -> Tensor:
        return self.encoder(stack_id)

    @torch.no_grad()
    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        anchor_embed = self(anchor_id)

        embeds = [self(stack_id) for stack_id in stack_ids]
        embeds = torch.cat(embeds, dim=0)

        scores = cosine_similarity(anchor_embed, embeds, dim=1)

        return scores.detach().cpu().tolist()
