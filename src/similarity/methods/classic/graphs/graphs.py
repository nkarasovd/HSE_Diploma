from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from src.similarity.methods.classic.tfidf import IntTfIdfComputer
from src.similarity.preprocess.seq_coder import SeqCoder


class GraphStructure(ABC):
    @abstractmethod
    def add_edge(self, edge: Tuple[int, ...], weight: Optional[float] = None):
        raise NotImplementedError

    @abstractmethod
    def get_size(self) -> int:
        raise NotImplementedError

    def build(self, edges: List[Tuple[int, ...]], weights: List[Optional[float]]):
        for edge, weight in zip(edges, weights):
            self.add_edge(edge, weight)


class EdgeList(GraphStructure):
    def __init__(self):
        self.edges = set()

    def add_edge(self, edge: Tuple[int, int], weight: Optional[float] = None):
        self.edges.add(edge)

    def get_size(self) -> int:
        return len(self.edges)

    def get_similarity(self, another_graph: 'EdgeList') -> float:
        common_edges_num = len(self.edges.intersection(another_graph.edges))
        return common_edges_num / min(self.get_size() + 1, another_graph.get_size() + 1)


class WeightedEdgeList(GraphStructure):
    def __init__(self):
        self.edges = dict()

    def add_edge(self, edge: Tuple[int, int], weight: Optional[float] = None):
        self.edges[edge] = self.edges.get(edge, 0) + weight

    def get_size(self) -> int:
        return sum(self.edges.values())

    def get_similarity(self, another_graph: 'WeightedEdgeList') -> float:
        self_edges = set(self.edges.keys())
        graph_edges = set(another_graph.edges.keys())

        common_edges = self_edges.intersection(graph_edges)
        edges_weight = sum(self.edges[edge] for edge in common_edges)

        # return edges_weight / min(self.get_size() + 1, another_graph.get_size() + 1)
        return edges_weight / max(self.get_size() + 1, another_graph.get_size() + 1)


class CrashGraphsWeighted:
    def __init__(self, coder: SeqCoder):
        self.coder = coder
        self.ns = (2,)

    def fit(self, sim_train_data: List[Tuple[int, int, int]] = None, stack_ids: List[int] = None):
        self.coder.fit(stack_ids)

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        group = WeightedEdgeList()
        for stack_id in stack_ids:
            edges = list(self.coder.ngrams(stack_id, ns=self.ns).keys())
            group.build(edges, [1 for _ in range(len(edges))])

        stack_trace = WeightedEdgeList()
        stack_trace_edges = list(self.coder.ngrams(anchor_id, ns=self.ns).keys())
        stack_trace.build(stack_trace_edges, [1 for _ in range(len(stack_trace_edges))])

        score = group.get_similarity(stack_trace)

        return [score] * len(stack_ids)


class TfIdfGraph:
    def __init__(self, coder: SeqCoder, ns: Tuple[int] = (1,)):
        self.tf_idf = IntTfIdfComputer(coder, ns)

    def fit(self, sim_train_data: List[Tuple[int, int, int]] = None,
            stack_ids: List[int] = None) -> 'TfIdfGraph':
        self.tf_idf.fit(stack_ids)
        return self

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        group = WeightedEdgeList()

        anchor_tfidf = self.tf_idf.transform(anchor_id)

        # scores = []
        edges, weights = [], []
        for stack_id in stack_ids:
            # score = 0
            for word in self.tf_idf.words_tfs(stack_id).keys():
                if word not in anchor_tfidf:
                    edges.append(word)
                    weights.append(0)
                    continue

                tf, idf = anchor_tfidf[word]
                tf_idf_pow2 = tf * idf ** 2
                edges.append(word)
                weights.append(tf_idf_pow2)
                # score += tf_idf_pow2

            # scores.append(score)
            group.build(edges, weights)

        # for stack_id in stack_ids:
        #     edges = list(self.coder.ngrams(stack_id, ns=self.ns).keys())
        #     group.build(edges, [1 for _ in range(len(edges))])

        stack_trace = WeightedEdgeList()
        stack_trace_edges = list(self.tf_idf.coder.ngrams(anchor_id, ns=self.tf_idf.ns).keys())
        stack_trace.build(stack_trace_edges, [1 for _ in range(len(stack_trace_edges))])

        score = group.get_similarity(stack_trace)

        return [score] * len(stack_ids)
