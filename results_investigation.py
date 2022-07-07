from ASTE.dataset.data.statistics import ResultInvestigator
from dataset.reader import ASTEDataset

from os import getcwd
from os.path import join

if __name__ == '__main__':
    for dataset in ['14lap']:
        label_path: str = join(getcwd(), 'dataset', 'data', 'ASTE_data_v2', dataset, 'test.txt')
        pred_path: str = join(getcwd(), 'results', '14lap_res.txt')

        original_data = ASTEDataset(label_path)
        predicted_data = ASTEDataset(pred_path)

        investigator = ResultInvestigator(original_data=original_data, model_prediction=predicted_data)
        investigator.compute()
        investigator.pprint()
        a, b, c = investigator.to_pandas()
        # print(a)



    # save_path: str = join(getcwd(), 'dataset', 'data', 'ASTE_statistics')
    # save_stats()
    #
    # # MULTIB #
    #
    # train_stats_comp = StatisticComparator()
    # dev_stats_comp = StatisticComparator()
    # test_stats_comp = StatisticComparator()
    #
    # for dataset in ['ca', 'eu']:
    #     data_path: str = join(getcwd(), 'dataset', 'data', 'multib', dataset)
    #     compute()
    #
    # save_path: str = join(getcwd(), 'dataset', 'data', 'multib_statistics')
    # save_stats()
