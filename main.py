from dataset.reader import DataReader
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

    dataset_reader = DataReader(data_path=data_path)
    dataset_reader.read('test.txt')
