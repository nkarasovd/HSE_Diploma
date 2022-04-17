from typing import Iterable, List, Dict, Optional

import faiss
import numpy as np
from torch import Tensor
from tqdm import tqdm

from src.similarity.data.buckets.event_state_model import StackAdditionState
from src.similarity.data.objects import Issue
from src.similarity.methods.dssm.model import DSSM
from src.similarity.methods.neural import device


class FaissFilter:
    def __init__(self, encoder: DSSM, num_stacks: int = 1000, num_issues: int = 20,
                 embed_dim: int = 64, num_candidates: Optional[int] = None):
        self.encoder = encoder
        self.vocab = dict()

        self.num_stacks = num_stacks
        self.num_issues = num_issues
        self.embed_dim = embed_dim
        self.num_candidates = num_candidates

        self.stack_ids = []
        self.index = None

        self.eval()

    def eval(self):
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.encoder.to(device)

    def encode(self, stack_id: int):
        return self.encoder(stack_id)

    def detach(self, embedding: Tensor) -> np.ndarray:
        return embedding.detach().cpu().flatten().numpy()

    def build_index(self, stack_ids: List[int]):
        vectors = [self.detach(self.encode(stack_id)) for stack_id in tqdm(stack_ids, desc="Build embeds")]
        vectors = np.array(vectors)

        faiss.normalize_L2(vectors)

        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(vectors)

        self.stack_ids = stack_ids
        self.num_candidates = self.num_candidates or len(stack_ids)

    def train_from_events(self, events: Iterable[StackAdditionState]) -> 'FaissFilter':
        stack_ids = []

        for i, event in tqdm(enumerate(events), desc="Handle events"):
            stack_ids.append(event.st_id)

            for id_, issue in event.issues.items():
                stack_ids.extend([stack.id for stack in issue.confident_state()])

        self.build_index(list(set(stack_ids)))

        return self

    def filter_top(self, event_id: int, anchor_id: int, issues: Dict[int, Issue]) -> Dict[int, List[int]]:
        query = self.detach(self.encode(anchor_id)).reshape(1, -1)
        faiss.normalize_L2(query)

        D, I = self.index.search(query, k=self.num_candidates)

        scores = {self.stack_ids[i]: d for i, d in zip(I[0], D[0])}

        issue_stack_dist = []
        for issue_id, issue in issues.items():
            stacks = [stack.id for stack in issue.confident_state()]
            top_stacks = list(filter(lambda x: x in scores, stacks))
            issue_stack_dist.extend([(issue_id, stack_id, scores[stack_id]) for stack_id in top_stacks])

        issue_stack_dist.sort(key=lambda x: x[2], reverse=True)

        top_issues = {}
        for i, (issue_id, stack_id, score) in enumerate(issue_stack_dist):
            top_issues[issue_id] = top_issues.get(issue_id, []) + [stack_id]

            if i > self.num_stacks and len(top_issues) > self.num_issues:
                return top_issues

        return top_issues
