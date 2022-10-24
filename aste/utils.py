import json
import logging
import os
from typing import Dict, List, Union

from envyaml import EnvYAML
from torch import Tensor


def ignore_index(func):
    def ignore_and_call(self, preds: Tensor, target: Tensor, *args, **kwargs):
        if self.ignore_index is not None:
            indices = (target != self.ignore_index).nonzero(as_tuple=False).flatten()
            preds_ignored = preds[indices]
            target_ignored = target[indices]
            if 'mask' in kwargs.keys():
                kwargs['mask'] = kwargs['mask'][indices]
            res = func(self, preds_ignored, target_ignored, *args, **kwargs)
        else:
            res = func(self, preds, target, *args, **kwargs)
        return res

    return ignore_and_call


def to_json(data_to_save: Union[Dict, List], path: str, mode: str = 'a') -> None:
    os.makedirs(path[:path.rfind(os.sep)], exist_ok=True)
    with open(path, mode=mode) as f:
        json.dump(data_to_save, f)


class ConfigReader:
    def __init__(self):
        pass

    @staticmethod
    def read_config(path: str) -> EnvYAML:
        cfg = EnvYAML(path)
        return cfg


try:
    config = ConfigReader.read_config(os.path.join(os.environ['CONFIG_FILE_PATH'], 'config.yml'))
except (FileNotFoundError, KeyError) as e:
    default: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml')
    config = ConfigReader.read_config(default)


def set_up_logger() -> None:
    logger = logging.getLogger()

    # Get handlers
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('logfile.log')

    # Log formatting
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # Set logging level
    if config["general"]["logging-level"] == 'DEBUG':
        logger.setLevel(logging.DEBUG)
        logging.getLogger('werkzeug').setLevel(logging.DEBUG)
    elif config["general"]["logging-level"] == 'WARNING':
        logger.setLevel(logging.WARNING)
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
        logging.getLogger('werkzeug').setLevel(logging.INFO)
