from typing import Dict, Optional

import wandb
from aste import utils

from aste.utils import config


class BaseTracker:
    def __init__(self, name: str = 'Base Tracker', tracker=None, *args, **kwargs):
        self.name: str = name
        self.tracker = tracker

    def log(self, logs) -> None:
        pass

    def add_config(self, new_config: Dict) -> None:
        pass

    def watch(self, model) -> None:
        pass

    def finish(self) -> None:
        pass

    def init(self, name: Optional[str] = None) -> None:
        pass


class WandbTracker(BaseTracker):
    def __init__(self, project: str, entity: str, *args, **kwargs):
        super().__init__('WandB Tracker', tracker=wandb)
        self.project: str = project
        self.entity: str = entity
        self.config: Dict = dict()
        not_relevant_keys = dict(config.environ()).keys()
        for key, value in dict(config).items():
            if key not in not_relevant_keys:
                self.config[key] = value

    def init(self, name: Optional[str] = None) -> None:
        self.tracker.init(project=self.project, entity=self.entity, config=self.config, name=name)

    def finish(self) -> None:
        self.tracker.finish()

    def add_config(self, new_config: Dict) -> None:
        utils.config.update(new_config)
        self.config.update(new_config)

    def log(self, logs: Dict) -> None:
        self.tracker.log(logs)

    def watch(self, model) -> None:
        self.tracker.watch(model)
