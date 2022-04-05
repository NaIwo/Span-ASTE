from dataset.reader import DatasetLoader
from utils import set_up_logger, config

import logging
import os


def log_introductory_info(data_path: str) -> None:
    logging.info(f"Data path: {data_path}")
    logging.info(f"Model type: {config['model']['type']}")
    logging.info(f"Encoder type: {config['encoder']['type']}")
    logging.info(f"Batch size: {config['dataset']['batch-size']}")


if __name__ == '__main__':
    data_path: str = os.path.join(os.getcwd(), 'dataset', 'data', 'ASTE_data_v2', '14lap')

    set_up_logger()
    log_introductory_info(data_path)

    dataset_reader = DatasetLoader(data_path=data_path)
    train = dataset_reader.load('train.txt')
    dev = dataset_reader.load('dev.txt')
    test = dataset_reader.load('test.txt')

