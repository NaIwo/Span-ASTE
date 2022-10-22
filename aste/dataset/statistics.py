import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Callable, Tuple, DefaultDict, Union, TypeVar

import pandas as pd
from tqdm import tqdm

from .domain import Sentence
from .reader import ASTEDataset
from .statistics_utils import (
    default_statistics_func,
    default_phrases_func,
    default_results_func,
    default_advanced_result_stats_func,
    StatsCounter)


def results_as_pandas(results, orient: str = 'index') -> pd.DataFrame:
    pd_results = pd.DataFrame.from_dict({(i, j): results[i][j]
                                         for i in results.keys()
                                         for j in results[i].keys()},
                                        orient=orient, dtype='float')
    return pd_results


class DatasetStatistics:
    def __init__(self, dataset: ASTEDataset,
                 statistics_func: Optional[Dict] = None,
                 phrases_func: Optional[Dict] = None,
                 phrases_to_count: Optional[List] = None):

        self.computed: bool = False

        self.dataset: ASTEDataset = dataset
        self.statistics_func: Dict = default_statistics_func if statistics_func is None else statistics_func
        self.statistics: Dict = {key: StatsCounter() for key in self.statistics_func.keys()}

        if phrases_to_count is None:
            self.phrases_to_count: List = ['not', "n't"]
            self.phrases_func: Dict = default_phrases_func if phrases_func is None else phrases_func
            key: str
            for key in self.phrases_func.keys():
                self.statistics[key] = {key: StatsCounter() for key in self.phrases_to_count}

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

    def to_csv(self, path: str) -> None:
        self.assert_compute()
        _, results = self.to_pandas()
        os.makedirs(path, exist_ok=True)
        results.to_csv(path)

    def to_pandas(self):
        key: str
        value: StatsCounter
        tmp_stats: Dict = {key: value.number() for key, value in self.statistics.items()}
        return pd.DataFrame(tmp_stats, index=[0])

    def to_json(self, path: str) -> None:
        self.assert_compute()
        os.makedirs(path[:path.rfind(os.sep)], exist_ok=True)
        with open(path, mode='w') as f:
            json.dump(self.statistics, f, indent=2, default=str)

    def assert_compute(self) -> None:
        assert self.computed, 'Firstly, you have to compute the statistics.'


SComp = TypeVar('SComp', bound='StatisticComparator')


class StatisticComparator:
    def __init__(self):
        self.flat_stats: DefaultDict = defaultdict(dict)
        self.nested_stats: DefaultDict = defaultdict(dict)

    @classmethod
    def from_dict(cls, statistics: Dict[str, Union[Dict, DatasetStatistics]]) -> SComp:
        obj: StatisticComparator = cls()
        obj.add(statistics=statistics)
        return obj

    def add(self, statistics: Dict[str, Union[Dict, DatasetStatistics]]) -> None:
        name: str
        statistic: Union[Dict, DatasetStatistics]
        for name, statistic in statistics.items():
            self.add_one_source(name, statistic)

    def add_one_source(self, source_name: str, statistic: Union[Dict, DatasetStatistics]) -> None:
        statistic: Dict = self.to_dict(statistic)
        stats_name: str
        result: Union[Dict, StatsCounter]
        for stats_name, result in tqdm(statistic.items(), desc=f'Processing source: {source_name}'):
            if self.is_flat(result):
                self.flat_stats[source_name].update({stats_name: result.number()})
            else:
                self.nested_stats[source_name].update({stats_name: self.nested_to_number(result)})

    @staticmethod
    def to_dict(statistic: Union[Dict, DatasetStatistics]) -> Dict:
        if isinstance(statistic, DatasetStatistics):
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
        with open(os.path.join(dir_path, 'flat_stats.json'), mode='w') as f:
            json.dump(flat_results_pd, f, indent=2, default=str)
        with open(os.path.join(dir_path, 'nested_stats.json'), mode='w') as f:
            json.dump(nested_results_pd, f, indent=2, default=str)

    def to_pandas(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        flat_results_pd: pd.DataFrame = pd.DataFrame.from_dict(self.flat_stats)
        nested_results_pd: pd.DataFrame = results_as_pandas(self.nested_stats)
        return flat_results_pd, nested_results_pd

    def pprint(self) -> None:
        print('Flat stats: ')
        print(json.dumps(self.flat_stats, indent=2, default=str))
        print('Nested stats: ')
        print(json.dumps(self.nested_stats, indent=2, default=str))


class ResultInvestigator:
    def __init__(self, original_data: ASTEDataset,
                 model_prediction: ASTEDataset,
                 result_stats_func: Optional[Dict] = None,
                 advanced_result_stats_func: Optional[Dict] = None,
                 statistics_func: Optional[Dict] = None,
                 phrases_func: Optional[Dict] = None,
                 phrases_to_count: Optional[List] = None):
        self.original_data_stats = DatasetStatistics(original_data, phrases_to_count=phrases_to_count,
                                                     statistics_func=statistics_func, phrases_func=phrases_func)
        self.model_prediction_stats = DatasetStatistics(model_prediction, phrases_to_count=phrases_to_count,
                                                        statistics_func=statistics_func, phrases_func=phrases_func)

        self.compared_results: Optional[StatisticComparator] = None

        self.model_prediction: ASTEDataset = model_prediction
        self.original_data: ASTEDataset = original_data

        self.result_stats_func: Dict = default_results_func if result_stats_func is None else result_stats_func
        self.result_stats: Dict = {key: StatsCounter() for key in self.result_stats_func.keys()}

        if advanced_result_stats_func is None:
            self.advanced_result_stats_func: Dict = default_advanced_result_stats_func
        else:
            self.advanced_result_stats_func: Dict = advanced_result_stats_func
        self.advanced_result_stats: Dict = {key: list() for key in self.advanced_result_stats_func.keys()}

    def compute(self) -> None:
        self.original_data_stats.compute()
        self.model_prediction_stats.compute()

        self.compared_results = StatisticComparator.from_dict({
            'original_labels': self.original_data_stats,
            'model_outputs': self.model_prediction_stats
        })

        self.compute_result_stats()

    def compute_result_stats(self) -> None:
        self.model_prediction.sort()
        self.original_data.sort()

        true: Sentence
        pred: Sentence
        for true, pred in tqdm(zip(self.original_data, self.model_prediction), desc=f'Result investigation...'):
            self.process_sentence(true, pred)
            self.process_sentence_advanced(true, pred)

        statistic_name: str
        results: List
        for statistic_name, results in self.advanced_result_stats.items():
            self.advanced_result_stats[statistic_name] = Counter(results)

    def process_sentence(self, true: Sentence, pred: Sentence) -> None:
        statistic_name: str
        func: Callable
        for statistic_name, func in self.result_stats_func.items():
            self.result_stats[statistic_name] += func(true, pred)

    def process_sentence_advanced(self, true: Sentence, pred: Sentence) -> None:
        statistic_name: str
        func: Callable
        for statistic_name, func in self.advanced_result_stats_func.items():
            self.advanced_result_stats[statistic_name] += func(true, pred)

    def pprint(self) -> None:
        self.assert_compute()
        self.compared_results.pprint()
        print(json.dumps(self.result_stats, indent=2, default=str))
        print(json.dumps(self.advanced_result_stats, indent=2, default=str))

    def to_csv(self, dir_path: str) -> None:
        self.assert_compute()
        self.compared_results.to_csv(dir_path)
        _, results, advanced_results = self.to_pandas()
        os.makedirs(dir_path, exist_ok=True)
        results.to_csv(os.path.join(dir_path, 'investigation_results.csv'))
        advanced_results.to_csv(os.path.join(dir_path, 'investigation_advanced_results.csv'))

    def to_json(self, dir_path: str) -> None:
        self.assert_compute()
        self.compared_results.to_json(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, 'investigation_results.json'), mode='w') as f:
            json.dump(self.result_stats, f, indent=2, default=str)
        with open(os.path.join(dir_path, 'investigation_advanced_results.json'), mode='w') as f:
            json.dump(self.advanced_result_stats, f, indent=2, default=str)

    def to_pandas(self):
        key: str
        value: Union[StatsCounter, List]
        tmp_stats: Dict = {key: value.number() if isinstance(value, StatsCounter) else value for key, value in
                           self.result_stats.items()}
        return self.compared_results.to_pandas(), pd.DataFrame(tmp_stats, index=[0]), results_as_pandas(
            self.advanced_result_stats)

    def assert_compute(self) -> None:
        assert self.compared_results is not None, 'Firstly, you have to compute the statistics.'
