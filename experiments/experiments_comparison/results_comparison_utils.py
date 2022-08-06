import json
import os

import pandas as pd
from os import listdir
from os.path import join, exists
from collections import defaultdict
from typing import List, Dict, DefaultDict

from ASTE.experiments.experiments_comparison.other_approach_results import OTHER_RESULTS
from ASTE.aste.utils import to_json

datasets: List = ['14lap', '14res', '15res', '16res']
AGG_DIR: str = 'endpoint'

score_results_file_name: str = 'final_results.json'
coverage_results_file_name: str = 'final_coverage_results.json'

results_path: str = join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiment_results', AGG_DIR)


def process(results_file_name, add_other_results: bool = False) -> Dict:
    results: Dict = dict()
    dataset: str
    for dataset in datasets:
        results[dataset] = process_single_dataset(join(results_path, dataset), results_file_name)
        if add_other_results:
            add_other_results_f(results, dataset)

    return results


def process_single_dataset(data_path: str, results_file_name: str) -> Dict:
    results: Dict = dict()
    dir_name: str
    for dir_name in listdir(data_path):
        res_path: str = join(data_path, dir_name, results_file_name)
        if not exists(res_path):
            continue
        with open(res_path, 'r') as f:
            results[dir_name] = to_percent(json.load(f))

    return results


def to_percent(results: Dict) -> Dict:
    return {k: v * 100 if 'Triplet' in k else v for k, v in results.items()}


def add_other_results_f(results: Dict, dataset_name: str) -> None:
    other_name: str
    for other_name in OTHER_RESULTS.keys():
        if dataset_name not in OTHER_RESULTS[other_name]:
            continue
        results[dataset_name][other_name] = OTHER_RESULTS[other_name][dataset_name]


def results_as_pandas(results, orient: str = 'index') -> pd.DataFrame:
    pd_results = pd.DataFrame.from_dict({(i, j): results[i][j]
                                         for i in results.keys()
                                         for j in results[i].keys()},
                                        orient=orient, dtype='float')
    return pd_results


def make_flatten_pd(results: pd.DataFrame) -> pd.DataFrame:
    results_dict: DefaultDict = results.to_dict()
    flatten_results = defaultdict(dict)
    dataset_name: str
    model_name: str
    score: float
    for (dataset_name, model_name), score in results_dict.items():
        flatten_results[dataset_name].update({model_name: score})

    return pd.DataFrame.from_dict(flatten_results).T


if __name__ == '__main__':
    # Scores as Dict
    scores: Dict = process(score_results_file_name, add_other_results=True)
    coverages: Dict = process(coverage_results_file_name)

    print(json.dumps(scores, indent=2))
    print(json.dumps(coverages, indent=2))

    # Scores as json
    to_json(scores, join(results_path, 'all_scores_results.json'), mode='w')
    to_json(coverages, join(results_path, 'all_coverage_results.json'), mode='w')

    # Scores as DataFrame
    pd_scores: pd.DataFrame = results_as_pandas(scores)
    pd_coverages: pd.DataFrame = results_as_pandas(coverages)

    pd_scores.to_csv(join(results_path, 'all_scores_results.csv'))
    pd_coverages.to_csv(join(results_path, 'all_coverage_results.csv'))

    print(pd_scores)
    print(pd_coverages)
