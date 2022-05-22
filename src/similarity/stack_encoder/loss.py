from typing import Optional

import torch
import torch.nn as nn
from torch import cosine_similarity

from src.similarity.data.buckets.event_state_model import StackAdditionState
from src.similarity.data.triplet_selector import TripletSelector
from src.similarity.methods.neural import device


class RankNetLossStackEncoder:
    def __init__(self, model, triplet_selector: TripletSelector):
        self.model = model
        self.loss_function = nn.CrossEntropyLoss()
        self.triplet_selector = triplet_selector
        self.sigmoid = nn.Sigmoid()

    def get_event_predictions(self, event: StackAdditionState):
        good_stacks, bad_stacks = self.triplet_selector(event)

        if len(good_stacks) == 0:
            return None

        predictions = []

        anchor_embed = self.model(event.st_id)

        for good_id, bad_id in zip(good_stacks, bad_stacks):
            good_sim = self.model(good_id)
            bad_sim = self.model(bad_id)

            good_pred = cosine_similarity(anchor_embed, good_sim)
            bad_pred = cosine_similarity(anchor_embed, bad_sim)

            out = self.sigmoid(good_pred - bad_pred)

            out_2 = torch.cat((1 - out, out), dim=0).view(-1, 2)

            predictions.append(out_2)

        return torch.cat(predictions)

    def get_event(self, event: StackAdditionState) -> Optional[torch.Tensor]:
        predictions = self.get_event_predictions(event)

        if predictions is None:
            return None

        target = torch.tensor([1] * len(predictions)).to(device)

        return self.loss_function(predictions, target)
