import copy
from random import shuffle
from typing import List, Iterable, Tuple, Dict, Union, Optional

import torch
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from tqdm import tqdm

from src.aggregation.data.objects import LabeledQueryFeatures
from src.aggregation.data.readers import read_data, read_features
from src.aggregation.data.scaler import Scaler
from src.aggregation.features.features_config import best_features, FeaturesConfig
from src.aggregation.scorers import PairwiseRankScorer
from src.aggregation.scorers.rank_model import LinearRankModel
from src.aggregation.train.train_log import BestResult
from src.aggregation.train.train_pairwise_scorer import train_pairwise_scorer, get_loss
from src.common.evaluation import paper_metrics_iter


def predict_by_features(model: PairwiseRankScorer,
                        labeled_query_features: Iterable[LabeledQueryFeatures],
                        scaler: Optional[StandardScaler] = None) -> Iterable[Tuple[int, Dict[int, float]]]:
    if scaler is None:
        for query in labeled_query_features:
            yield query.right_issue_id, model.score_by_features(query.issues_features)
    else:
        for query in labeled_query_features:
            query = build_scaled_queries(scaler, [query])[0]
            yield query.right_issue_id, model.score_by_features(query.issues_features)


def build_scaled_queries(scaler, queries: List[LabeledQueryFeatures]) -> List[LabeledQueryFeatures]:
    scaled_queries = []

    for q in queries:
        d = {i: scaler.transform([j])[0] for i, j in q.issues_features.items()}
        scaled_queries.append(LabeledQueryFeatures(q.right_issue_id, d))

    return scaled_queries


def transform_queries(train_queries: List[LabeledQueryFeatures],
                      test_queries: Optional[List[LabeledQueryFeatures]]) -> Tuple[List[LabeledQueryFeatures],
                                                                                   Union[List[LabeledQueryFeatures],
                                                                                         StandardScaler]]:
    scaler = StandardScaler()
    features = []
    for q in train_queries:
        features += list(q.issues_features.values())

    scaler.fit(features)

    train_queries = build_scaled_queries(scaler, train_queries)
    if test_queries is None:
        return train_queries, scaler

    test_queries = build_scaled_queries(scaler, test_queries)
    return train_queries, test_queries


def dump_weights(weights: List[float], weights_dump_path: Optional[str]):
    with open(weights_dump_path, 'w') as f:
        for i in range(0, len(weights), 1):
            f.write(str(weights[i]) + '\n')


def fit_by_features(train_queries: List[LabeledQueryFeatures],
                    test_queries: Union[List[LabeledQueryFeatures], str],
                    features_config: FeaturesConfig,
                    num_epochs: int = 250, jb: bool = True,
                    scale_queries: bool = False,
                    weights_dump_path: Optional[str] = None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    rank_model = LinearRankModel(features_config.get_config_dim()).to(device)
    optimizer = Adam(rank_model.parameters())

    if scale_queries and jb:
        # support only our data
        train_queries, test_queries = transform_queries(train_queries, test_queries)
    elif scale_queries and not jb:
        train_queries, scaler = transform_queries(train_queries, None)

    evaluation_scorer = PairwiseRankScorer(rank_model, features_config, Scaler(), device)

    best_result = None

    for epoch in tqdm(range(num_epochs)):
        rank_model.train()
        shuffle(train_queries)

        for train_query in train_queries:
            optimizer.zero_grad()
            loss = get_loss(rank_model, train_query, device, 10)
            if loss is not None:
                loss.backward()
                optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'\n\nEpoch = {epoch + 1}')
            rank_model.eval()
            if jb:
                scores = paper_metrics_iter(predict_by_features(evaluation_scorer, test_queries))
            else:
                test_generator = read_features(test_queries, train_mode=False, disable=True)
                if scale_queries:
                    scores = paper_metrics_iter(predict_by_features(evaluation_scorer, test_generator, scaler))
                else:
                    scores = paper_metrics_iter(predict_by_features(evaluation_scorer, test_generator))

            print("\nWeights\n")
            rank_model.print_info()
            weights = rank_model.model[0].weight.data[0].tolist()
            dump_weights(weights, weights_dump_path + f"epoch_{epoch}.txt")

            # Save new best result
            if best_result is None or scores['rr@1'][0] > best_result.scores['rr@1'][0]:
                best_result = BestResult(epoch, copy.deepcopy(rank_model), scores)

    best_result.print_log()

    # if weights_dump_path is not None:
    #     rank_model = best_result.rank_model
    #     weights = rank_model.model[0].weight.data[0].tolist()
    #
    #     dump_weights(weights, weights_dump_path)


def base_experiment(train_path: str, test_path: str,
                    num_epochs: int = 750, scale: bool = False):
    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}\n")

    q_tr = read_data(train_path, train_mode=True)
    q_te = read_data(test_path, train_mode=False)

    train_pairwise_scorer(q_tr, q_te, best_features, num_epochs, scale)


def features_experiment(train_path: str, test_path: str,
                        num_epochs: int = 750, jb: bool = True,
                        scale_queries: bool = False,
                        weights_dump_path: Optional[str] = None):
    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}\n")

    q_tr = list(read_features(train_path, train_mode=True))
    if jb:
        q_te = list(read_features(test_path, train_mode=False))
        fit_by_features(q_tr, q_te, best_features, num_epochs,
                        weights_dump_path=weights_dump_path,
                        scale_queries=scale_queries)
    else:
        fit_by_features(q_tr, test_path, best_features, num_epochs,
                        jb=jb, scale_queries=scale_queries,
                        weights_dump_path=weights_dump_path)


if __name__ == '__main__':
    # base_experiment("/home/centos/karasov/diploma/jb_dumps/agg_train/lerch.json",
    #                 "/home/centos/karasov/diploma/jb_dumps/agg_test/lerch.json", num_epochs=100)

    # features_experiment("/home/centos/karasov/diploma/jb_features_dumps/agg_train/lerch.json",
    #                     "/home/centos/karasov/diploma/jb_features_dumps/agg_test/lerch.json",
    #                     num_epochs=500, scale_queries=True,
    #                     weights_dump_path="./jb_weights/scaled_lerch_")

    features_experiment("/home/centos/karasov/diploma/netbeans_features_dumps/agg_train/levenshtein.json",
                        "/home/centos/karasov/diploma/netbeans_features_dumps/agg_test/levenshtein.json",
                        jb=False, num_epochs=250, scale_queries=True,
                        weights_dump_path="./netbeans_weights/scaled_levenshtein/")
