import os
import json
from collections import defaultdict
from typing import Dict, DefaultDict

from ASTE.aste.utils import to_json

SAVE_DIR: str = 'hard'


def aggregate(path: str) -> None:
    num_of_metric_files: int = 0
    num_of_coverage_files: int = 0
    results: DefaultDict = defaultdict(float)
    coverage_results: DefaultDict = defaultdict(float)
    file_name: str
    for file_name in os.listdir(path):
        if 'metrics_results' in file_name:
            num_of_metric_files += 1
            file_path = os.path.join(path, file_name)
            with open(file_path) as json_file:
                result: Dict = json.load(json_file)['triplet_metric']
            results['TripletPrecision'] += result['TripletPrecision']
            results['TripletRecall'] += result['TripletRecall']
            results['TripletF1'] += result['TripletF1']

        elif 'coverage_results' in file_name:
            num_of_coverage_files += 1
            file_path = os.path.join(path, file_name)
            with open(file_path) as json_file:
                result: Dict = json.load(json_file)
            coverage_results['Ratio'] += result['Ratio']
            coverage_results['Extracted spans'] += result['Extracted spans']
            coverage_results['Total correct spans'] += result['Total correct spans']

    for key, value in results.items():
        results[key] = value / num_of_metric_files

    for key, value in coverage_results.items():
        coverage_results[key] = value / num_of_coverage_files

    to_json(results, os.path.join(path, 'final_results.json'), mode='w')
    to_json(coverage_results, os.path.join(path, 'final_coverage_results.json'), mode='w')

    print(results)
    print(coverage_results)


if __name__ == '__main__':
    data_path: str = os.path.join(os.getcwd(), 'experiments', 'experiment_results')

    dataset_name: str
    for dataset_name in ['14lap', '14res', '15res', '16res', 'ca', 'eu']:
        print(dataset_name)
        path: str = os.path.join(data_path, dataset_name, SAVE_DIR)
        aggregate(path)
