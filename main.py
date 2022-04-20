from dataset.reader import DatasetLoader
from utils import set_up_logger, config
from ASTE.aste.trainer import Trainer
from ASTE.aste.models import BaseModel, BertBaseModel
from ASTE.dataset.encoders import BertEncoder
from ASTE.aste.trainer_tools import WandbTracker, BaseTracker

import logging
from typing import Dict
import os


def log_introductory_info(data_path: str) -> None:
    logging.info(f"Data path: {data_path}")
    logging.info(f"Batch size: {config['dataset']['batch-size']}")


if __name__ == '__main__':
    data_path: str = os.path.join(os.getcwd(), 'dataset', 'data', 'ASTE_data_v2', '14lap')
    set_up_logger()

    dataset_reader = DatasetLoader(data_path=data_path)
    train_data = dataset_reader.load('train.txt', encoder=BertEncoder())
    dev_data = dataset_reader.load('dev.txt', encoder=BertEncoder())
    test_data = dataset_reader.load('test.txt', encoder=BertEncoder())

    model: BaseModel = BertBaseModel()

    tracker: BaseTracker = WandbTracker(project="...", entity="...")
    tracker.init()
    log_introductory_info(data_path)

    trainer: Trainer = Trainer(model=model, tracker=tracker)
    trainer.train(train_data=train_data, dev_data=dev_data)
    # trainer.load_model(os.path.join(os.getcwd(), 'results', '14lap-span_range-0.8', 'bert_base_model.pth'))
    trainer.load_model(trainer.save_path)
    threshold_results: Dict = dict()
    prediction_threshold: int
    for prediction_threshold in [0.3, 0.4, 0.45, 0.4999, 0.5, 0.55]:
        trainer.set_prediction_threshold(prediction_threshold)
        results: Dict = trainer.check_coverage_detected_spans(test_data)
        threshold_results[prediction_threshold] = results

    plot_data = [[label, val] for label, val in threshold_results.items()]
    table = trainer.tracker.tracker.Table(data=plot_data, columns=["threshold", "results"])
    tracker.tracker.log({"Threshold results": table})
