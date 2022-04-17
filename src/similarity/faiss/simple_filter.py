from typing import List, Dict, Iterable

import torch
from torch import Tensor
from torch import cosine_similarity
from tqdm import tqdm

from src.similarity.data.buckets.event_state_model import StackAdditionState
from src.similarity.data.objects import Issue
from src.similarity.methods.dssm.model import DSSM
from src.similarity.methods.neural import device


class SimpleFilter:
    def __init__(self, encoder: DSSM, num_stacks: int = 1000, num_issues: int = 20):
        self.encoder = encoder
        self.vocab = dict()

        self.num_stacks = num_stacks
        self.num_issues = num_issues

        self.eval()

    def eval(self):
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.encoder.to(device)

    def encode(self, stack_id: int) -> Tensor:
        return self.encoder(stack_id)

    def build_vocab(self, stack_ids: List[int]):
        self.vocab = {stack_id: self.encode(stack_id) for stack_id in tqdm(stack_ids, desc="Build embeds")}

    def get_embed(self, stack_id: int) -> Tensor:
        return self.vocab[stack_id]

    def score(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        anchor_embed = self.get_embed(anchor_id)

        embeds = [self.get_embed(stack_id) for stack_id in stack_ids]
        embeds = torch.cat(embeds, dim=0)

        return cosine_similarity(anchor_embed, embeds).detach().cpu().tolist()

    def filter_top(self, event_id: int, anchor_id: int, issues: Dict[int, Issue]) -> Dict[int, List[int]]:
        issue_stack_dist = []
        for id_, issue in issues.items():
            stacks = [stack.id for stack in issue.confident_state()]

            if stacks:
                scores = self.score(anchor_id, stacks)
                issue_stack_dist.extend([(id_, stacks[i], scores[i]) for i in range(len(stacks))])

        issue_stack_dist.sort(key=lambda x: x[2], reverse=True)

        top_issues = dict()
        for i, el in enumerate(issue_stack_dist):
            issue_id, stack_id, score = el
            top_issues[issue_id] = top_issues.get(issue_id, [])
            top_issues[issue_id].append(stack_id)

            if i > self.num_stacks and len(top_issues) > self.num_issues:
                return top_issues

        return top_issues

    def train_from_events(self, events: Iterable[StackAdditionState]):
        stack_ids = []

        for i, event in tqdm(enumerate(events), desc="Events"):
            stack_ids.append(event.st_id)

            for id_, issue in event.issues.items():
                stack_ids.extend([stack.id for stack in issue.confident_state()])

        self.build_vocab(list(set(stack_ids)))
