from dataset.reader import DatasetLoader
from utils import set_up_logger, config
from ASTE.aste.trainer import Trainer
from ASTE.aste.models import BaseModel, BertBaseModel
from ASTE.aste.tools import WandbTracker, BaseTracker
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric
from ASTE.dataset.domain.sentence import Sentence

import logging
from typing import Dict
import os


def log_introductory_info(data_path: str) -> None:
    logging.info(f"Data path: {data_path}")
    logging.info(f"Batch size: {config['dataset']['batch-size']}")
    logging.info(f"Effective batch size: {config['dataset']['effective-batch-size']}")


if __name__ == '__main__':

    # Select a dataset and set appropriate paths
    dataset_name: str = '14lap'
    data_path: str = os.path.join(os.getcwd(), 'dataset', 'data', 'ASTE_data_v2', dataset_name)

    # Setup logger properties
    set_up_logger()

    # Create dataset reader. Path should point out directory with data - NOT EXACT DATA FILE
    dataset_reader = DatasetLoader(data_path=data_path)

    # Load data. Here, you should point to extract file with data.
    train_data = dataset_reader.load('train.txt')
    dev_data = dataset_reader.load('dev.txt')
    test_data = dataset_reader.load('test.txt')

    # Get Model. You can create your own definition or update existing one.
    model: BaseModel = BertBaseModel()

    # Setup experiment tracker. Default is BaseTracker (it means: do not track the experiment.)
    tracker: BaseTracker = WandbTracker(project="project_name", entity="entity")
    # If you want set experiment name, you can init tracker by hand as below. If don't, tracker will be init by Trainer.
    tracker.init(name=dataset_name)
    # You can pass your own information and any other configuration which you want to keep.
    tracker.add_config({'Dataset': dataset_name})

    # You can log information about Dataset and batch.
    log_introductory_info(data_path)

    # You can provide you own save path or let Trainer pick up one for you.
    save_path: str = os.path.join(os.getcwd(), 'results', 'name', 'bert_base_model.pth')
    # Define trainer - Wrapper for model handling.
    trainer: Trainer = Trainer(model=model, tracker=tracker, save_path=save_path)

    # TRAIN MODEL
    trainer.train(train_data=train_data, dev_data=dev_data)

    # LOAD BEST MODEL
    trainer.load_model(save_path)

    # You can check coverage of correct detected spans
    trainer.check_coverage_detected_spans(test_data)
    # Or you can test your model on selected dataset
    results: Dict = trainer.test(test_data)

    # You can also save the results. Just type:
    save_path_json: str = os.path.join(os.getcwd(), 'results', 'name', 'results.json')
    results[ModelMetric.NAME].to_json(path=save_path_json)

    # If you want to feed the model with your own sentence. You should create this sentence, and call predict:
    sentence = Sentence('This OS is so fast !!')
    prediction: ModelOutput = trainer.predict(sentence)
    # You can also save the results to file
    prediction.save('sentence_result.txt')



