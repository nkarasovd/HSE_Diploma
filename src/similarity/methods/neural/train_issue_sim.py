from typing import List, Tuple, Iterable

import torch
from tqdm import tqdm

from src.common.evaluation import paper_metrics_iter
from src.similarity.data.buckets.bucket_data import DataSegment, BucketData
from src.similarity.data.buckets.event_state_model import StackAdditionState
from src.similarity.data.triplet_selector import RandomTripletSelector
from src.similarity.evaluation.issue_sim import score_model
from src.similarity.methods.hypothesis_selection import HypothesisSelector
from src.similarity.methods.neural.losses import LossComputer, RanknetLossComputer
from src.similarity.methods.neural.neural_base import NeuralModel
from src.similarity.methods.pair_stack_issue_model import PairStackBasedSimModel, MaxIssueScorer


def log_metrics(sim_stack_model: NeuralModel, filter_model: HypothesisSelector, loss_computer: LossComputer,

                train_sim_pairs_data_for_score: List[Tuple[int, int, int]],
                test_sim_pairs_data_for_score: List[Tuple[int, int, int]],

                train_data_for_score: List[StackAdditionState],
                test_data_for_score: List[StackAdditionState],

                prefix: str, writer, n_iter: int):
    with torch.no_grad():
        train_loss_value = loss_computer.get_eval_raws(train_sim_pairs_data_for_score)
        test_loss_value = loss_computer.get_eval_raws(test_sim_pairs_data_for_score)

        ps_model = PairStackBasedSimModel(sim_stack_model, MaxIssueScorer(), filter_model)
        train_preds = ps_model.predict(train_data_for_score)
        test_preds = ps_model.predict(test_data_for_score)
        train_score = score_model(train_preds, full=False)
        test_score = score_model(test_preds, full=False)
    print(prefix +
          f"Train loss: {round(train_loss_value, 4)}. "
          f"Test loss: {round(test_loss_value, 4)}. "
          f"Train prec {train_score[0]}, rec {train_score[1]}. "
          f"Test prec {test_score[0]}, rec {test_score[1]}       ")  # , end=''

    if writer:
        writer.add_scalar('Loss/train', train_loss_value, n_iter)
        writer.add_scalar('Loss/test', test_loss_value, n_iter)


def log_all_data_scores(sim_stack_model: NeuralModel, filter_model: HypothesisSelector,
                        test_events: Iterable[StackAdditionState]):
    ps_model = PairStackBasedSimModel(sim_stack_model, MaxIssueScorer(), filter_model)  # =None for no filter

    test_preds = ps_model.predict(test_events)
    print("Test")
    paper_metrics_iter(test_preds)

    print()


def train_issue_model(sim_stack_model: NeuralModel, data: BucketData, train: DataSegment, test: DataSegment,
                      loss_name: str, optimizers: List, filter_model: HypothesisSelector = None,
                      epochs: int = 1, batch_size: int = 25, selection_from_event_num: int = 4,
                      writer=None, period: int = 25):
    # if loss_name == "point":
    #     if filter_model is None:
    #         train_selector = RandomPairSimSelector(selection_from_event_num)
    #     else:
    #         train_selector = TopPairSimSelector(filter_model, selection_from_event_num)
    #     loss_computer = PointLossComputer(sim_stack_model, train_selector)
    # elif loss_name == "ranknet":
    #     if filter_model is None:
    #         train_selector = RandomTripletSelector(selection_from_event_num)
    #     else:
    #         train_selector = TopTripletSelector(filter_model, selection_from_event_num)
    #     loss_computer = RanknetLossComputer(sim_stack_model, train_selector)
    # elif loss_name == "triplet":
    #     if filter_model is None:
    #         train_selector = RandomTripletSelector(selection_from_event_num)
    #     else:
    #         train_selector = TopTripletSelector(filter_model, selection_from_event_num)
    #     loss_computer = TripletLossComputer(sim_stack_model, train_selector, margin=0.2)
    # else:
    #     raise ValueError

    train_selector = RandomTripletSelector(selection_from_event_num)
    loss_computer = RanknetLossComputer(sim_stack_model, train_selector)

    for epoch in range(epochs):
        for i, event in tqdm(enumerate(data.get_events(train)), desc="Train S3M"):
            sim_stack_model.train()
            loss = loss_computer.get_event(event)
            if loss is None:
                continue

            loss /= batch_size
            loss.backward()

            if (i + 1) % batch_size == 0:
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()

        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            sim_stack_model.eval()
            log_all_data_scores(sim_stack_model, filter_model, data.get_events(test))

        print()
        print(f"Epoch {epoch} done.")

    return sim_stack_model
