from typing import Iterable

from tqdm import tqdm

from src.common.evaluation import paper_metrics_iter
from src.similarity.data.buckets.bucket_data import DataSegment, BucketData
from src.similarity.data.buckets.event_state_model import StackAdditionState
from src.similarity.methods.hypothesis_selection import HypothesisSelector
from src.similarity.methods.pair_stack_issue_model import PairStackBasedSimModel, MaxIssueScorer
from src.similarity.stack_encoder.loss import RankNetLossStackEncoder
from src.similarity.stack_encoder.model import StackEncoder
from src.similarity.stack_encoder.selector import StackEncoderSelector


def log_all_data_scores(sim_stack_model: StackEncoder, filter_model: HypothesisSelector,
                        test_events: Iterable[StackAdditionState]):
    ps_model = PairStackBasedSimModel(sim_stack_model, MaxIssueScorer(), filter_model)  # =None for no filter

    test_preds = ps_model.predict(test_events)
    print("Test")
    paper_metrics_iter(test_preds)


def fit_dssm_model(sim_stack_model: StackEncoder, data: BucketData, train: DataSegment, test: DataSegment,
                   optimizer, filter_model: HypothesisSelector = None,
                   epochs: int = 1, batch_size: int = 25, selection_from_event_num: int = 5):
    train_selector = StackEncoderSelector(selection_from_event_num, selection_from_event_num)

    loss_computer = RankNetLossStackEncoder(sim_stack_model, train_selector)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch}.")

        sim_stack_model.train()
        for i, event in tqdm(enumerate(data.get_events(train)), leave=True, position=0, desc="Train DSSM"):
            loss = loss_computer.get_event(event)
            if loss is None:
                continue

            loss /= batch_size
            loss.backward()

            if (i + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 1 == 0:
            sim_stack_model.eval()
            log_all_data_scores(sim_stack_model, filter_model, data.get_events(test))

    return sim_stack_model
