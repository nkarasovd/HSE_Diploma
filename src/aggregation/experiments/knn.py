from typing import Any, Dict

from src.aggregation.data.readers import read_data
from src.aggregation.evaluation.prediction import predict
from src.aggregation.scorers.knn.distance_based import AdjustedAveragingMethod, AdjustedWeightingMethod, \
    TruncatedPotentialsMethod
from src.aggregation.scorers.knn.k_conditional_nn import KConditionalNN
from src.aggregation.scorers.knn.kernel_methods import UniformKernelKNN, TriangleKernelKNN, \
    EpanechnikovKernelKNN, QuarticKernelKNN, TriweightKernelKNN, GaussianKernelKNN, CosineKernelKNN, \
    TricubeKernelKNN, LogisticKernelKNN, SigmoidKernelKNN, SilvermanKernelKNN
from src.aggregation.utils import timeit
from src.common.evaluation import paper_metrics_iter


@timeit
def kernel_methods(data_path: str, verbose: bool = False) -> Dict[str, Any]:
    print(f"Data path: {data_path}")
    data = read_data(data_path)

    kernel_scorers = [UniformKernelKNN, TriangleKernelKNN, EpanechnikovKernelKNN,
                      QuarticKernelKNN, TriweightKernelKNN, GaussianKernelKNN,
                      CosineKernelKNN, TricubeKernelKNN, LogisticKernelKNN,
                      SigmoidKernelKNN, SilvermanKernelKNN]

    best_results = dict()
    for scorer in kernel_scorers:
        best_results[scorer.__name__] = None
        print(scorer.__name__)
        for k in range(1, 16, 2):
            if verbose:
                print(f"k = {k}")
            predictions = predict(scorer(k), data)
            scores = paper_metrics_iter(predictions, verbose=verbose)

            if best_results[scorer.__name__] is None or \
                    scores['rr@1'][0] > best_results[scorer.__name__][1]['rr@1'][0]:
                best_results[scorer.__name__] = (k, scores)

    return best_results


@timeit
def k_conditional_scorer(data_path: str, verbose: bool = False) -> Dict[str, Any]:
    print(f"Data path: {data_path}")
    data = read_data(data_path)

    best_results = dict()
    best_results[KConditionalNN.__name__] = None

    print(KConditionalNN.__name__)
    for k in range(1, 16, 2):
        if verbose:
            print(f"k = {k}")
        predictions = predict(KConditionalNN(k), data)
        scores = paper_metrics_iter(predictions, verbose=verbose)

        if best_results[KConditionalNN.__name__] is None or \
                scores['rr@1'][0] > best_results[KConditionalNN.__name__][1]['rr@1'][0]:
            best_results[KConditionalNN.__name__] = (k, scores)

    return best_results


@timeit
def distance_based(data_path: str, verbose: bool = False) -> Dict[str, Any]:
    print(f"Data path: {data_path}")
    data = read_data(data_path)

    best_results = dict()
    best_results[AdjustedAveragingMethod.__name__] = None
    for alpha in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  10, 20, 30, 40, 50, 60, 70, 80, 90,
                  100, 110, 120, 130, 140, 150, 200, 250, 300, 500]:
        if verbose:
            print(f"alpha = {alpha}")
        predictions = predict(AdjustedAveragingMethod(alpha), data)
        scores = paper_metrics_iter(predictions, verbose=verbose)
        if best_results[AdjustedAveragingMethod.__name__] is None or \
                scores['rr@1'][0] > best_results[AdjustedAveragingMethod.__name__][1]['rr@1'][0]:
            best_results[AdjustedAveragingMethod.__name__] = (alpha, scores)
        print()

    best_results[AdjustedWeightingMethod.__name__] = None
    for omega in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
        if verbose:
            print(f"omega = {omega}")

        for gamma in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      20, 30, 40, 50, 60, 70, 80, 90, 100,
                      200, 300, 400, 500, 600, 700, 800, 900, 1000,
                      1500, 2000, 3000, 5000]:
            if verbose:
                print(f"gamma = {gamma}")
            predictions = predict(AdjustedWeightingMethod(omega, gamma), data)
            scores = paper_metrics_iter(predictions, verbose=verbose)
            if best_results[AdjustedWeightingMethod.__name__] is None or \
                    scores['rr@1'][0] > best_results[AdjustedWeightingMethod.__name__][1]['rr@1'][0]:
                best_results[AdjustedWeightingMethod.__name__] = ((omega, gamma), scores)
            print()

    best_results[TruncatedPotentialsMethod.__name__] = None
    for beta in [0, 1, 2, 3, 4, 5, 10, 25, 50, 100, 150, 250]:
        if verbose:
            print(f"beta = {beta}")
        predictions = predict(TruncatedPotentialsMethod(beta), data)
        scores = paper_metrics_iter(predictions, verbose=verbose)
        if best_results[TruncatedPotentialsMethod.__name__] is None or \
                scores['rr@1'][0] > best_results[TruncatedPotentialsMethod.__name__][1]['rr@1'][0]:
            best_results[TruncatedPotentialsMethod.__name__] = (beta, scores)
        print()

    return best_results


if __name__ == '__main__':
    # path = "/home/centos/karasov/diploma/jb_dumps/sim_test/tracesim.json"
    path = "/home/centos/karasov/diploma/netbeans_dumps/sim_test/s3m.json"
    results = kernel_methods(path, verbose=False)

    for k, v in results.items():
        print(k)
        params, scores = v[0], v[1]
        print(f"Params: {params}")
        for name, score in scores.items():
            print(f"{name}: {score}")
