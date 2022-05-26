import os
import json
from collections import defaultdict
from typing import Dict, DefaultDict


def aggregate(path: str) -> None:
    num_of_files: int = 0
    results: DefaultDict = defaultdict(float)
    file_name: str
    for file_name in os.listdir(path):
        if 'metrics_results' not in file_name:
            continue
        num_of_files += 1
        file_path = os.path.join(path, file_name)
        with open(file_path) as json_file:
            result: Dict = json.load(json_file)['triplet_metric']
        results['TripletPrecision'] += result['TripletPrecision']
        results['TripletRecall'] += result['TripletRecall']
        results['TripletF1'] += result['TripletF1']
    for key, value in results.items():
        results[key] = value / num_of_files

    save_path: str = os.path.join(path, 'final_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f)

    print(results)


if __name__ == '__main__':
    data_path: str = os.path.join(os.getcwd(), 'experiment_results')

    dataset_name: str
    for dataset_name in ['14lap', '14res', '15res', '16res']:
        print(dataset_name)
        path: str = os.path.join(data_path, dataset_name)
        aggregate(path)
