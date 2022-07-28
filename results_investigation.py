from ASTE.dataset.data.statistics import ResultInvestigator
from dataset.reader import ASTEDataset

from os import getcwd
from os.path import join

if __name__ == '__main__':
    # You can perform investigation about your results by using ResultInvestigator class.
    for dataset in ['14lap', '14res', '15res', '16res']:
        label_path: str = join(getcwd(), 'dataset', 'data', 'ASTE_data_v2', dataset, 'test.txt')
        pred_path: str = join(getcwd(), 'results', f'{dataset}_res.txt')

        # One thing is required to do. Build ASTEDataset from original data and predicted one
        original_data = ASTEDataset(label_path)
        predicted_data = ASTEDataset(pred_path)

        # Pass them to class constructor
        # NOTE: You can pass your own implementation functions to calculate more stats:
        #   ResultInvestigator(
        #                  model_prediction: ASTEDataset,
        #                  result_stats_func: Optional[Dict] = None,
        #                  advanced_result_stats_func: Optional[Dict] = None,
        #                  statistics_func: Optional[Dict] = None,
        #                  phrases_func: Optional[Dict] = None,
        #                  phrases_to_count: Optional[List] = None)
        investigator = ResultInvestigator(original_data=original_data, model_prediction=predicted_data)
        # Now you can compute the result and save to json or csv as well.
        investigator.compute()
        investigator.pprint()
        investigator.to_pandas()
