from ASTE.dataset.reader import ASTEDataset
from ASTE.dataset.domain.sentence import Sentence
from ASTE.dataset.data.statistics_utils import default_statistics_func, default_phrases_func, StatsCounter
from ASTE.experiments.experiments_comparison.results_comparison_utils import results_as_pandas

import os
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Optional, Callable, Tuple, DefaultDict, Union


class DatasetStatistic:
    def __init__(self, dataset: ASTEDataset,
                 phrases_to_count: Optional[List] = None,
                 statistics_func: Optional[Dict] = None,
                 phrases_func: Optional[Dict] = None):

        self.computed: bool = False

        self.dataset: ASTEDataset = dataset
        self.statistics_func: Dict = default_statistics_func if statistics_func is None else statistics_func
        self.statistics: Dict = dict.fromkeys(self.statistics_func.keys(), StatsCounter())

        if phrases_to_count is None:
            self.phrases_to_count: List = ['not', "n't"]
            self.phrases_func: Dict = default_phrases_func if phrases_func is None else phrases_func
            key: str
            for key in self.phrases_func.keys():
                self.statistics[key] = dict.fromkeys(self.phrases_to_count, StatsCounter())

    def compute(self) -> Dict:
        self.computed = True
        sentence: Sentence
        for sentence in tqdm(self.dataset, desc=f'Statistics calculation...'):
            self.process_sentence(sentence)

        return self.statistics

    def process_sentence(self, sentence: Sentence) -> None:
        statistic_name: str
        func: Callable
        for statistic_name, func in self.statistics_func.items():
            self.statistics[statistic_name] += func(sentence)

        if self.phrases_to_count:
            for phrase in self.phrases_to_count:
                self.process_phrase(sentence, phrase)

    def process_phrase(self, sentence: Sentence, phrase: str) -> None:
        name: str
        func: Callable
        for name, func in self.phrases_func.items():
            self.statistics[name][phrase] += func(sentence, phrase)

    def pprint(self) -> None:
        self.assert_compute()
        print(json.dumps(self.statistics, indent=2, default=str))

    def to_json(self, path: str) -> None:
        self.assert_compute()
        os.makedirs(path[:path.rfind(os.sep)], exist_ok=True)
        with open(path, mode='w') as f:
            json.dump(self.statistics, f, indent=2, default=str)

    def assert_compute(self) -> None:
        assert self.computed, 'Firstly, you have to compute the statistics.'


class StatisticComparison:
    def __init__(self):
        self.flat_stats: DefaultDict = defaultdict(dict)
        self.nested_stats: DefaultDict = defaultdict(dict)

    def add(self, statistics: Dict[str, Union[Dict, DatasetStatistic]]) -> None:
        name: str
        statistic: Union[Dict, DatasetStatistic]
        for name, statistic in statistics.items():
            self.add_one_source(name, statistic)

    def add_one_source(self, source_name: str, statistic: Union[Dict, DatasetStatistic]) -> None:
        statistic: Dict = self.to_dict(statistic)
        stats_name: str
        result: Union[Dict, StatsCounter]
        for stats_name, result in statistic.items():
            if self.is_flat(result):
                self.flat_stats[source_name].update({stats_name: result.number()})
            else:
                self.nested_stats[source_name].update({stats_name: self.nested_to_number(result)})

    @staticmethod
    def to_dict(statistic: Union[Dict, DatasetStatistic]) -> Dict:
        if isinstance(statistic, DatasetStatistic):
            return statistic.statistics
        else:
            return statistic

    @staticmethod
    def is_flat(statistic: Union[Dict, StatsCounter]) -> bool:
        if isinstance(statistic, StatsCounter):
            return True
        else:
            return False

    @staticmethod
    def nested_to_number(statistic: Dict) -> Dict:
        key: str
        value: StatsCounter
        return {key: value.number() for key, value in statistic.items()}

    def to_csv(self, dir_path: str) -> None:
        flat_results_pd, nested_results_pd = self.to_pandas()
        os.makedirs(dir_path, exist_ok=True)
        flat_results_pd.to_csv(os.path.join(dir_path, 'flat_stats.csv'))
        nested_results_pd.to_csv(os.path.join(dir_path, 'nested_stats.csv'))

    def to_json(self, dir_path: str) -> None:
        flat_results_pd, nested_results_pd = self.to_pandas()
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, 'flat_stats.csv'), mode='w') as f:
            json.dump(flat_results_pd, f, indent=2, default=str)
        with open(os.path.join(dir_path, 'nested_stats.csv'), mode='w') as f:
            json.dump(nested_results_pd, f, indent=2, default=str)

    def to_pandas(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        flat_results_pd: pd.DataFrame = pd.DataFrame.from_dict(self.flat_stats)
        nested_results_pd: pd.DataFrame = results_as_pandas(self.nested_stats)
        return flat_results_pd, nested_results_pd
