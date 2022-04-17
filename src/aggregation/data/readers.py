import json
from typing import List, Iterable

from tqdm import tqdm

from src.aggregation.data.objects import DistInfo, IssueDists, Query, LabeledQuery, LabeledQueryFeatures


def read_data(filename: str, size: int = 25000, train_mode: bool = False) -> List[LabeledQuery]:
    res = []
    with open(filename) as f:
        for i, line in enumerate(tqdm(f, position=0, leave=True, desc="Read dump file")):
            if i >= size:
                break
            event = json.loads(line)

            issues_scores, issues = dict(), []
            for k, v in event["issues"].items():
                # keys in json are strings
                issues_scores[int(k)] = v
                issue_dists = tuple(DistInfo(float(dist), hist[0], hist[1]) for dist, hist in v.items())
                issues.append(IssueDists(int(k), issue_dists))

            query = LabeledQuery(event["right"], Query(tuple(issues)))
            if train_mode:
                if event["right"] in issues_scores and len(issues_scores[event["right"]]) != 0:
                    res.append(query)
            else:
                res.append(query)
    return res


def read_features(filename: str, size: int = 25000, train_mode: bool = False,
                  disable: bool = False) -> Iterable[LabeledQueryFeatures]:
    with open(filename) as f:
        for i, line in enumerate(tqdm(f, disable=disable, position=0,
                                      leave=True, desc="Read dump file")):
            if i >= size:
                break
            event = json.loads(line)

            issues_features, issues = dict(), []
            for k, v in event["issues"].items():
                # keys in json are strings
                issues_features[int(k)] = v

            query = LabeledQueryFeatures(event["right"], issues_features)
            if train_mode:
                if event["right"] in issues_features and len(issues_features[event["right"]]) != 0:
                    yield query
            else:
                yield query
