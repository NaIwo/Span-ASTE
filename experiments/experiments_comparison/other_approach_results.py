from typing import Dict

# https://github.com/NJUNLP/GTS#results=, ASTE-Data-V2-dataset
# https://arxiv.org/pdf/2010.04640.pdf
gts: Dict = {'info': f'Results taken from: https://arxiv.org/pdf/2010.04640.pdf',
             '14lap': {'TripletPrecision': 58.54, 'TripletRecall': 50.65, 'TripletF1': 54.3},
             '14res': {'TripletPrecision': 68.71, 'TripletRecall': 67.67, 'TripletF1': 68.17},
             '15res': {'TripletPrecision': 60.69, 'TripletRecall': 60.54, 'TripletF1': 60.61},
             '16res': {'TripletPrecision': 67.39, 'TripletRecall': 66.73, 'TripletF1': 67.06}}

# https://aclanthology.org/2021.acl-long.367.pdf, ASTE-Data-V2-dataset
span_based: Dict = {'info': 'Results taken from: https://aclanthology.org/2021.acl-long.367.pdf',
                    '14lap': {'TripletPrecision': 63.44, 'TripletRecall': 55.84, 'TripletF1': 59.38},
                    '14res': {'TripletPrecision': 72.89, 'TripletRecall': 70.89, 'TripletF1': 71.85},
                    '15res': {'TripletPrecision': 62.18, 'TripletRecall': 64.45, 'TripletF1': 63.27},
                    '16res': {'TripletPrecision': 69.45, 'TripletRecall': 71.17, 'TripletF1': 70.26}}

# https://arxiv.org/pdf/2102.08549v3.pdf, ASTE-Data-V2-dataset
first_to_then_polarity: Dict = {'info': 'https://arxiv.org/pdf/2102.08549v3.pdf',
                                '14lap': {'TripletPrecision': 57.84, 'TripletRecall': 59.33, 'TripletF1': 58.58},
                                '14res': {'TripletPrecision': 63.59, 'TripletRecall': 73.44, 'TripletF1': 68.16},
                                '15res': {'TripletPrecision': 54.53, 'TripletRecall': 63.3, 'TripletF1': 58.59},
                                '16res': {'TripletPrecision': 63.57, 'TripletRecall': 71.98, 'TripletF1': 67.52}}

# https://aclanthology.org/2021.acl-short.64.pdf, ASTE-Data-V2-dataset
t5: Dict = {'info': 'https://aclanthology.org/2021.acl-short.64.pdf',
            '14lap': {'TripletPrecision': None, 'TripletRecall': None, 'TripletF1': 60.78},
            '14res': {'TripletPrecision': None, 'TripletRecall': None, 'TripletF1': 72.16},
            '15res': {'TripletPrecision': None, 'TripletRecall': None, 'TripletF1': 62.1},
            '16res': {'TripletPrecision': None, 'TripletRecall': None, 'TripletF1': 70.1}}

# https://arxiv.org/pdf/2010.02609.pdf, ASTE-Data-V2-dataset
jet_t: Dict = {'info': 'https://arxiv.org/pdf/2010.02609.pdf',
               '14lap': {'TripletPrecision': 53.53, 'TripletRecall': 43.28, 'TripletF1': 47.86},
               '14res': {'TripletPrecision': 63.44, 'TripletRecall': 54.12, 'TripletF1': 58.41},
               '15res': {'TripletPrecision': 68.2, 'TripletRecall': 42.89, 'TripletF1': 52.66},
               '16res': {'TripletPrecision': 65.28, 'TripletRecall': 51.95, 'TripletF1': 57.85}}

# https://arxiv.org/pdf/2010.02609.pdf, ASTE-Data-V2-dataset
jet_o: Dict = {'info': 'https://arxiv.org/pdf/2010.02609.pdf',
               '14lap': {'TripletPrecision': 55.39, 'TripletRecall': 47.33, 'TripletF1': 51.04},
               '14res': {'TripletPrecision': 70.56, 'TripletRecall': 55.94, 'TripletF1': 62.4},
               '15res': {'TripletPrecision': 64.45, 'TripletRecall': 51.96, 'TripletF1': 57.53},
               '16res': {'TripletPrecision': 70.42, 'TripletRecall': 58.37, 'TripletF1': 63.83}}

# https://arxiv.org/pdf/2103.15255.pdf, ASTE-Data-V2-dataset
more_fine_grained: Dict = {'info': 'https://arxiv.org/pdf/2103.15255.pdf',
                           '14lap': {'TripletPrecision': 56.6, 'TripletRecall': 55.1, 'TripletF1': 55.8},
                           '14res': {'TripletPrecision': 69.3, 'TripletRecall': 69.0, 'TripletF1': 69.2},
                           '15res': {'TripletPrecision': 55.8, 'TripletRecall': 61.5, 'TripletF1': 58.5},
                           '16res': {'TripletPrecision': 61.2, 'TripletRecall': 72.7, 'TripletF1': 66.5}}

# https://openreview.net/pdf?id=Z9vIuaFlIXx, ASTE-Data-V2-dataset
sambert: Dict = {'info': 'https://openreview.net/pdf?id=Z9vIuaFlIXx',
                 '14lap': {'TripletPrecision': 62.26, 'TripletRecall': 59.15, 'TripletF1': 60.66},
                 '14res': {'TripletPrecision': 70.29, 'TripletRecall': 74.92, 'TripletF1': 72.53},
                 '15res': {'TripletPrecision': 65.12, 'TripletRecall': 63.51, 'TripletF1': 64.3},
                 '16res': {'TripletPrecision': 68.01, 'TripletRecall': 75.44, 'TripletF1': 71.53}}

# https://arxiv.org/pdf/2204.12674.pdf, ASTE-Data-V2-dataset
SBC: Dict = {'info': 'https://arxiv.org/pdf/2204.12674.pdf',
                 '14lap': {'TripletPrecision': 63.64, 'TripletRecall': 61.80, 'TripletF1': 62.71},
                 '14res': {'TripletPrecision': 77.09, 'TripletRecall': 70.99, 'TripletF1': 73.92},
                 '15res': {'TripletPrecision': 63.00, 'TripletRecall': 64.95, 'TripletF1': 63.96},
                 '16res': {'TripletPrecision': 75.20, 'TripletRecall': 71.40, 'TripletF1': 73.25}}

OTHER_RESULTS: Dict = {
    'gts': gts,
    'span_based': span_based,
    'first_to_then_polarity': first_to_then_polarity,
    't5': t5,
    'jet_t': jet_t,
    'jet_o': jet_o,
    'more_fine_grained': more_fine_grained,
    'sambert': sambert,
    'SBC': SBC,
}
