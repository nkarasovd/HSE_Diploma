from src.aggregation.data.readers import read_data
from src.aggregation.evaluation.prediction import predict
from src.aggregation.scorers.max_scorer import MaxScorer
from src.aggregation.utils import timeit
from src.common.evaluation import paper_metrics_iter


@timeit
def max_scorer(data_path: str):
    print(f"Data path: {data_path}\n")
    predictions = predict(MaxScorer(), read_data(data_path))
    paper_metrics_iter(predictions)


if __name__ == '__main__':
    max_scorer("/home/centos/karasov/diploma/netbeans_dumps/sim_test/lerch.json")
