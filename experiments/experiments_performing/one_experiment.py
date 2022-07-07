from ASTE.dataset.reader import DatasetLoader
from ASTE.utils import set_up_logger, config
from ASTE.aste.trainer import Trainer
from ASTE.aste.models import BaseModel, BertBaseModel
from ASTE.aste.tools import WandbTracker, BaseTracker
from ASTE.aste.models import ModelMetric, ModelLoss, ModelOutput
from ASTE.aste.utils import to_json

import os
import logging
import argparse
from typing import Dict, List
from torch.cuda import empty_cache

NUM_EXPERIMENTS: int = 7


def log_introductory_info() -> None:
    logging.info(f"Data path: {data_path}")
    logging.info(f"Batch size: {config['dataset']['batch-size']}")
    logging.info(f"Effective batch size: {config['dataset']['effective-batch-size']}")
    logging.info(f'Experiment number: {experiment_idx}')


def run() -> None:
    dev_score: float = 0.0
    while dev_score <= 1e-5:
        dataset_reader = DatasetLoader(data_path=data_path)
        train_data = dataset_reader.load('train.txt')
        dev_data = dataset_reader.load('dev.txt')
        test_data = dataset_reader.load('test.txt')

        tracker: BaseTracker = BaseTracker(project="...", entity="...")
        tracker.init(name=f'{dataset_name}_{experiment_idx}')
        tracker.add_config({'Dataset': dataset_name,
                            'Experiment number': experiment_idx})
        log_introductory_info()

        trainer: Trainer = Trainer(model=BertBaseModel(), tracker=tracker, save_path=save_path)

        trainer.train(train_data=train_data, dev_data=dev_data)

        trainer.load_model(save_path)

        local_results: Dict = trainer.test(dev_data)
        dev_score = local_results[ModelMetric.NAME].triplet_metric['TripletF1']

    local_results: Dict = trainer.test(test_data)
    coverage_results: Dict = trainer.check_coverage_detected_spans(test_data)
    to_json(data_to_save=coverage_results, path=coverage_save_path)

    local_results[ModelMetric.NAME].to_json(path=metric_save_path)
    local_results[ModelLoss.NAME].to_json(path=loss_save_path)

    local_results: List[ModelOutput] = trainer.predict(test_data)
    ModelOutput.save_list_of_outputs(local_results, model_output_save_path)

    del trainer
    del tracker
    del dataset_reader
    del train_data
    del dev_data
    del test_data
    empty_cache()


def arg_parse():
    parser = argparse.ArgumentParser(description='Information for the experiment.')
    parser.add_argument('--dataset_name', '-d', type=str, help='Name of dataset.', required=True)
    parser.add_argument('--agg_type', '-agg', type=str, help='Name of aggregation type.', required=False,
                        default='sum_aggregation')
    parser.add_argument('--save_dir_name', '-s', type=str, help='Name of save directory.', default='')
    parser.add_argument('--id', '-id', type=int, help='Experiment id.', required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arg = arg_parse()
    set_up_logger()
    dataset_name: str = arg.dataset_name
    agg_type: str = arg.agg_type
    save_dir_name: str = arg.save_dir_name
    experiment_idx: int = arg.id
    data_path: str = os.path.join(os.getcwd(), 'dataset', 'data', 'ASTE_data_v2', dataset_name)
    save_path: str = os.path.join(os.getcwd(), 'experiments', 'experiment_results', agg_type, f'{dataset_name}',
                                  save_dir_name, f'model.pth')
    metric_save_path: str = os.path.join(os.getcwd(), 'experiments', 'experiment_results', agg_type, f'{dataset_name}',
                                         save_dir_name, f'metrics_results_{experiment_idx}.json')
    loss_save_path: str = os.path.join(os.getcwd(), 'experiments', 'experiment_results', agg_type, f'{dataset_name}',
                                       save_dir_name, f'losses_results_{experiment_idx}.json')
    coverage_save_path: str = os.path.join(os.getcwd(), 'experiments', 'experiment_results', agg_type,
                                           f'{dataset_name}', save_dir_name, f'coverage_results_{experiment_idx}.json')
    model_output_save_path: str = os.path.join(os.getcwd(), 'experiments', 'experiment_results', agg_type,
                                               f'{dataset_name}', save_dir_name, f'model_output_{experiment_idx}.txt')

    # RUN EXPERIMENTS
    run()
