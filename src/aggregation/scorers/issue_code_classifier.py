from typing import Any, Dict

from src.aggregation.data.objects import Query
from src.aggregation.scorers import AggregationScorer


class IssueCodeClassifier(AggregationScorer):
    def __init__(self, clf: AggregationScorer):
        self.clf = clf

    @staticmethod
    def load(config: Dict[str, Any]) -> 'AggregationScorer':
        pass

    def save(self, model_path: str):
        pass

    def score(self, query: Query) -> Dict[int, float]:
        base_scores = self.clf.score(query)

        score_issues = dict()
        for issue, score in base_scores.items():
            score_issues[score] = score_issues.get(score, []) + [issue]

        scores = dict()
        shift = 0
        for score, issues in dict(sorted(score_issues.items())).items():
            for issue in sorted(issues):
                scores[issue] = score + shift
                shift += 1

        return scores
