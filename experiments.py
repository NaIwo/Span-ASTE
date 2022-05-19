from dataset.reader import DatasetLoader
from utils import set_up_logger, config
from ASTE.aste.trainer import Trainer
from ASTE.aste.models import BaseModel, BertBaseModel
from ASTE.aste.tools import WandbTracker, BaseTracker
from ASTE.aste.models import ModelMetric, ModelLoss

import os
import logging
import argparse
from typing import Dict
from torch.cuda import empty_cache


def log_introductory_info() -> None:
    logging.info(f"Data path: {data_path}")
    logging.info(f"Batch size: {config['dataset']['batch-size']}")
    logging.info(f"Effective batch size: {config['dataset']['effective-batch-size']}")
    logging.info(f'Experiment number: {experiment_idx}')


def run() -> None:
    dataset_reader = DatasetLoader(data_path=data_path)
    train_data = dataset_reader.load('train.txt')
    dev_data = dataset_reader.load('dev.txt')
    test_data = dataset_reader.load('test.txt')

    tracker: BaseTracker = WandbTracker(project="name", entity="entity")
    tracker.init(name=f'{dataset_name}_{experiment_idx}')
    tracker.add_config({'Dataset': dataset_name,
                        'Experiment number': experiment_idx})
    log_introductory_info()

    trainer: Trainer = Trainer(model=BertBaseModel(), tracker=tracker, save_path=save_path)

    trainer.train(train_data=train_data, dev_data=dev_data)

    trainer.load_model(save_path)
    local_results: Dict = trainer.test(test_data)

    local_results[ModelMetric.NAME].to_json(path=metric_save_path)
    local_results[ModelLoss.NAME].to_json(path=loss_save_path)

    del trainer
    del tracker
    del dataset_reader
    del train_data
    del dev_data
    del test_data
    del local_results
    empty_cache()


def arg_parse():
    parser = argparse.ArgumentParser(description='Information for the experiment.')
    parser.add_argument('--dataset_name', '-d', type=str, help='Name of dataset.', required=True)
    parser.add_argument('--id', '-id', type=int, help='Experiment id.', required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arg = arg_parse()
    set_up_logger()
    dataset_name: str = arg.dataset_name
    experiment_idx: int = arg.id
    data_path: str = os.path.join(os.getcwd(), 'dataset', 'data', 'ASTE_data_v2', dataset_name)
    save_path: str = os.path.join(os.getcwd(), 'experiment_results', f'{dataset_name}',
                                  f'model_{experiment_idx}.pth')
    metric_save_path: str = os.path.join(os.getcwd(), 'experiment_results', f'{dataset_name}',
                                         f'metrics_results_{experiment_idx}.json')
    loss_save_path: str = os.path.join(os.getcwd(), 'experiment_results', f'{dataset_name}',
                                       f'losses_results_{experiment_idx}.json')

    # RUN EXPERIMENTS
    run()
