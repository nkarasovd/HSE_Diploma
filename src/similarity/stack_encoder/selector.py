from typing import List, Tuple

import numpy as np

from src.similarity.data.buckets.event_state_model import StackAdditionState
from src.similarity.data.triplet_selector import TripletSelector


class StackEncoderSelector(TripletSelector):
    def __init__(self, size: int = None, num_good_stacks: int = 1):
        self.size = size
        self.num_good_stacks = num_good_stacks

    def __call__(self, event: StackAdditionState) -> Tuple[List[int], List[int]]:
        """
        Return list of good stack ids and bad stack ids

        :param issues: dict of issue_id to issue
        :param is_id: the issue new stack was attached
        :return: list of good stack ids and bad stack ids
        """
        if event.is_id not in event.issues:
            return [], []

        good_stacks = np.array(list(event.issues[event.is_id].stacks.keys()))
        bad_stacks = []

        for iid in event.issues.keys() - {event.is_id}:
            bad_stacks += list(event.issues[iid].stacks.keys())

        good_stacks = good_stacks[np.random.choice(len(good_stacks), self.num_good_stacks)]
        bad_stacks = np.array(bad_stacks)[np.random.choice(len(bad_stacks), self.size)]

        return list(good_stacks), list(bad_stacks)
