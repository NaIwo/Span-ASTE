from ASTE.aste.models.base_model import BaseModel
from ASTE.utils import config
from ASTE.dataset.domain.const import ChunkCode

import torch
from torch.utils.data import DataLoader
from typing import Optional
import logging
from datetime import datetime
from tqdm import tqdm


class Trainer:
    def __init__(self, model: BaseModel):
        logging.info(f"Model '{model.model_name}' has been initialized.")
        self.model: BaseModel = model.to(config['general']['device'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['model']['learning-rate'])
        self.chunk_loss = torch.nn.CrossEntropyLoss(ignore_index=int(ChunkCode.NOT_RELEVANT))

    def train(self, train_data: DataLoader, dev_data: Optional[DataLoader] = None):
        self.model.train()
        training_start_time: datetime.time = datetime.now()
        logging.info(f'Training start at time: {training_start_time}')
        for epoch in range(config['model']['epochs']):
            logging.info(f'Epoch: {epoch}')
            epoch_loss: float = 0.
            for batch in tqdm(train_data):
                sentence, chunk_label, mask = batch
                model_out: torch.Tensor = self.model(sentence, mask)
                loss = self.chunk_loss(model_out, chunk_label)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += loss
            logging.info(f'Epoch loss: {epoch_loss}')
        training_stop_time: datetime.time = datetime.now()
        logging.info(f'Training stop at time: {training_stop_time}')
        logging.info(f'Training time in seconds: {(training_stop_time - training_start_time).seconds}')
