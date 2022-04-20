from ASTE.utils import config

import wandb
from typing import Dict


class BaseTracker:
    def __init__(self, name: str = 'Base Tracker', tracker=None):
        self.name: str = name
        self.tracker = tracker

    def log(self, logs) -> None:
        pass

    def add_config(self, new_config: Dict) -> None:
        pass

    def watch(self, model) -> None:
        pass

    def init(self) -> None:
        pass


class WandbTracker(BaseTracker):
    def __init__(self, project: str, entity: str):
        super().__init__('WandB Tracker', tracker=wandb)
        self.project: str = project
        self.entity: str = entity
        self.config: Dict = dict()
        not_relevant_keys = dict(config.environ()).keys()
        for key, value in dict(config).items():
            if key not in not_relevant_keys:
                self.config[key] = value

    def init(self) -> None:
        self.tracker.init(project=self.project, entity=self.entity, config=self.config)

    def add_config(self, new_config: Dict) -> None:
        self.tracker.config.update(new_config)
        self.config.update(new_config)

    def log(self, logs: Dict) -> None:
        self.tracker.log(logs)

    def watch(self, model) -> None:
        self.tracker.watch(model)
