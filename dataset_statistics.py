from aste.dataset.statistics import DatasetStatistics, StatisticComparator
from aste.dataset.reader import ASTEDataset

from os import getcwd
from os.path import join


def compute():
    for data_source in ['train.txt', 'dev.txt', 'test.txt']:

        # You have to create ASTE Dataset to compute all of statistics
        aste_dataset = ASTEDataset(join(data_path, data_source))

        # NOTE: You can define your own method and new statistics and pass them to DatasetStatistic class.
        # You can also pass words to search
        # Possible fields: dataset: ASTEDataset,
        #                  phrases_to_count: Optional[List] = None,
        #                  statistics_func: Optional[Dict] = None,
        #                  phrases_func: Optional[Dict] = None
        # Default (and example as well) statistics_func and phrases_func
        # can be found in 'ASTE/dataset/statistics/statistics_utils/defaults.py' file

        # Now, only one thing left to do: pass dataset and call 'compute'
        dataset_stats = DatasetStatistics(dataset=aste_dataset)
        dataset_stats.compute()

        # If you want, you can pretty print the results
        dataset_stats.pprint()

        # Or add results to comparison
        if 'train' in data_source:
            # One possible way to do that is pass dict of stats.
            # You can pass many key: value pairs, for example {14lap: stats, 14res: stats} or
            # {14lap_train: train_stats, 14lap_dev: dev_stats} and so on...
            train_stats_comp.add({dataset: dataset_stats})
        elif 'dev' in data_source:
            # Another way is to pass single result as in line below
            dev_stats_comp.add_one_source(source_name=dataset, statistic=dataset_stats)
        elif 'test' in data_source:
            # Last, you can pass your own stats dict (dataset_stats.statistics is a dict of results)
            test_stats_comp.add({dataset: dataset_stats.statistics})


def save_stats():
    stats: StatisticComparator
    for stats, name in zip([train_stats_comp, dev_stats_comp, test_stats_comp], ['train', 'dev', 'test']):
        # You can save the results to csv (or json by calling 'to_json').
        # NOTE: pass path to directory, without file name.
        stats.to_csv(join(save_path, name))


if __name__ == '__main__':
    # You can compare statistics in different level of depth.
    # In this case we keep stats for train, dev and test sets separately

    train_stats_comp = StatisticComparator()
    dev_stats_comp = StatisticComparator()
    test_stats_comp = StatisticComparator()

    for dataset in ['14lap', '14res', '15res', '16res']:
        data_path: str = join(getcwd(), 'dataset', 'data', 'ASTE_data_v2', dataset)
        compute()

    save_path: str = join(getcwd(), 'dataset', 'data', 'ASTE_statistics')
    save_stats()

    # MULTIB #

    train_stats_comp = StatisticComparator()
    dev_stats_comp = StatisticComparator()
    test_stats_comp = StatisticComparator()

    for dataset in ['ca', 'eu']:
        data_path: str = join(getcwd(), 'dataset', 'data', 'multib', dataset)
        compute()

    save_path: str = join(getcwd(), 'dataset', 'data', 'multib_statistics')
    save_stats()
