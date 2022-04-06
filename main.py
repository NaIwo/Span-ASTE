import torch

from dataset.reader import DatasetLoader
from utils import set_up_logger, config
from ASTE.aste.trainer import Trainer
from ASTE.aste.models import BaseModel, BertBaseModel
from ASTE.dataset.encoders import BertEncoder

import logging
import os


def log_introductory_info(data_path: str, model_type: str) -> None:
    logging.info(f"Data path: {data_path}")
    logging.info(f"Model type: {model_type}")
    logging.info(f"Batch size: {config['dataset']['batch-size']}")


if __name__ == '__main__':
    data_path: str = os.path.join(os.getcwd(), 'dataset', 'data', 'ASTE_data_v2', '14lap')
    set_up_logger()

    model: BaseModel = BertBaseModel()

    log_introductory_info(data_path, model.model_name)

    dataset_reader = DatasetLoader(data_path=data_path)
    train_data = dataset_reader.load('train.txt', encoder=BertEncoder())
    dev_data = dataset_reader.load('dev.txt', encoder=BertEncoder())
    test_data = dataset_reader.load('test.txt', encoder=BertEncoder())

    trainer: Trainer = Trainer(model=model)
    trainer.train(train_data=train_data, dev_data=dev_data)
    # trainer.load_model(os.path.join(os.getcwd(), 'results', '20220406-192236', 'bert_base_model.pth'))
    # sentence, chunk_label, mask = next(iter(test_data))
    # print(torch.argmax(trainer.predict(sentence, mask), dim=-1))
    # print(chunk_label)

