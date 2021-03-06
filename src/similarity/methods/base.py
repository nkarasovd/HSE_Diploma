from abc import abstractmethod, ABC
from typing import Tuple, List, Dict, Iterable, Union

from src.similarity.data.objects import StackEvent
from src.similarity.data.buckets.bucket_data import BucketData, DataSegment


class SimStackModel(ABC):
    @abstractmethod
    def fit(self, sim_train_data: List[Tuple[int, int, int]] = None, unsup_data: List[int] = None) -> 'SimStackModel':
        raise NotImplementedError

    def find_params(self, sim_val_data: List[Tuple[int, int, int]]) -> 'SimStackModel':
        return self

    @abstractmethod
    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        raise NotImplementedError

    def predict_pairs(self, sim_data: List[Tuple[int, int, int]]) -> List[float]:
        return [self.predict(st_id1, [st_id2])[0] for st_id1, st_id2, l in sim_data]

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class SimIssueModel(ABC):
    def find_params(self, data: BucketData, val: DataSegment) -> 'SimIssueModel':
        return self

    @abstractmethod
    def predict(self, events: Iterable[StackEvent]) -> List[Tuple[int, Dict[int, float]]]:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class IssueScorer:
    @abstractmethod
    def score(self, scores: Iterable[float], with_arg: bool = False) -> Union[float, Tuple[float, int]]:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
