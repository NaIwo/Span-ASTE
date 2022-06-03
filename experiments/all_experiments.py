from ASTE.dataset.reader import DatasetLoader
from ASTE.utils import set_up_logger, config
from ASTE.aste.trainer import Trainer
from ASTE.aste.models import BaseModel, BertBaseModel
from ASTE.aste.tools import WandbTracker, BaseTracker
from ASTE.aste.models import ModelMetric, ModelLoss
from ASTE.aste.utils import to_json

import os
import json
import logging
from typing import Dict
from torch.cuda import empty_cache
from collections import defaultdict

NUM_EXPERIMENTS: int = 7
SAVE_DIR_NAME: str = ''


def log_introductory_info() -> None:
    logging.info(f"Data path: {data_path}")
    logging.info(f"Batch size: {config['dataset']['batch-size']}")
    logging.info(f"Effective batch size: {config['dataset']['effective-batch-size']}")
    logging.info(f'Experiment number: {experiment_idx}')


def run() -> Dict:
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

    del trainer
    del tracker
    del dataset_reader
    del train_data
    del dev_data
    del test_data
    empty_cache()

    return local_results


if __name__ == '__main__':
    set_up_logger()
    final_metrics_results: Dict = dict()
    for dataset_name in ['14lap', '14res', '15res', '16res']:
        final_metrics_results[dataset_name] = defaultdict(float)
        experiment_idx: int
        for experiment_idx in range(NUM_EXPERIMENTS):
            data_path: str = os.path.join(os.getcwd(), 'dataset', 'data', 'ASTE_data_v2', dataset_name)
            save_path: str = os.path.join(os.getcwd(), 'experiments', 'experiment_results', f'{dataset_name}',
                                          SAVE_DIR_NAME, f'model.pth')
            metric_save_path: str = os.path.join(os.getcwd(), 'experiments', 'experiment_results', f'{dataset_name}',
                                                 SAVE_DIR_NAME, f'metrics_results_{experiment_idx}.json')
            loss_save_path: str = os.path.join(os.getcwd(), 'experiments', 'experiment_results', f'{dataset_name}',
                                               SAVE_DIR_NAME, f'losses_results_{experiment_idx}.json')
            coverage_save_path: str = os.path.join(os.getcwd(), 'experiments', 'experiment_results', f'{dataset_name}',
                                                   SAVE_DIR_NAME, f'coverage_results_{experiment_idx}.json')

            # RUN EXPERIMENTS
            results: Dict = run()

            metric_name: str
            score: float
            for metric_name, score in results[ModelMetric.NAME].triplet_metric.items():
                if 'Triplet' in metric_name:
                    final_metrics_results[dataset_name][metric_name] += score

        for metric_name, score in final_metrics_results[dataset_name].items():
            final_metrics_results[dataset_name][metric_name] = score / NUM_EXPERIMENTS

    final_results: str = os.path.join(os.getcwd(), 'experiments', 'experiment_results', f'{dataset_name}',
                                      SAVE_DIR_NAME, f'final_results.json')
    with open(final_results, 'w') as f:
        json.dump(final_results, f)
