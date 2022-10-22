import json
from os.path import join
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, DefaultDict, Tuple

from aste.utils import to_json

SEP: str = '\\'

ERRORS: DefaultDict = defaultdict(int)


def convert_dataset() -> None:
    for data_source in ['train.json', 'dev.json', 'test.json']:
        data_path: str = join('.', dataset_name, data_source)
        with open(data_path, 'r') as file:
            data: List = json.load(file)

        convert_sentences(data, data_path)


def convert_sentences(data: List, data_path: str) -> None:
    save_path: str = data_path.replace('.json', '_bio.json')
    results: List = list()

    row: Dict
    for row in data:
        sentence: Dict = dict()
        sentence['sentence'] = row['text']
        sentence['triples'] = list()

        splitted_sentence = row['text'].split()
        splitted_sentence = list(map(lambda el: f'{el}{SEP}O', splitted_sentence))

        triplets: List = get_triplets_from_row(row)

        if not triplets:
            continue
        triplet: Tuple
        for triplet in triplets:
            target_tags = deepcopy(splitted_sentence)
            start: int = triplet[0][0]
            end: int = triplet[0][1] if len(triplet[0]) > 1 else start
            for idx in range(start, end + 1):
                tag: str = 'I'
                if idx == start:
                    tag: str = 'B'
                target_tags[idx] = target_tags[idx].replace(f'{SEP}O', f'{SEP}{tag}')

            opinion_tags = deepcopy(splitted_sentence)
            start: int = triplet[1][0]
            end: int = triplet[1][1] if len(triplet[1]) > 1 else start
            for idx in range(start, end + 1):
                tag: str = 'I'
                if idx == start:
                    tag: str = 'B'
                opinion_tags[idx] = opinion_tags[idx].replace(f'{SEP}O', f'{SEP}{tag}')

            temp: Dict = {
                'target_tags': ' '.join(target_tags),
                'opinion_tags': ' '.join(opinion_tags),
                'sentiment': triplet[-1]
            }
            sentence['triples'].append(temp)
        results.append(sentence)

    to_json(results, save_path, mode='w')


def get_triplets_from_row(row: Dict) -> List:
    sentence = row['text']
    words_map: Dict = get_words_map(sentence)

    triplets: List = list()
    element: Dict
    for element in row['opinions']:
        if len(element['Target'][1]) == 0 or len(element['Polar_expression'][1]) == 0:
            continue
        try:
            aspect = element['Target'][1][0].split(':')
            splitted_words = sentence[int(aspect[0]): int(aspect[1])].split()
            aspect = [words_map[int(aspect[0])], words_map[int(aspect[1])]] if len(
                splitted_words) > 1 else [words_map[int(aspect[0])]]

            opinion = element['Polar_expression'][1][0].split(':')
            splitted_words = sentence[int(opinion[0]): int(opinion[1])].split()
            opinion = [words_map[int(opinion[0])], words_map[int(opinion[1])]] if len(
                splitted_words) > 1 else [words_map[int(opinion[0])]]

            sentiment = element['Polarity']

            triplets.append((aspect, opinion, sentiment.lower()))

        except Exception as e:
            ERRORS[dataset_name] += 1

    return triplets


def get_words_map(sentence: str) -> Dict:
    words_map: Dict = dict()
    word_idx: int = 0
    letter_idx: int
    letter: str
    for letter_idx, letter in enumerate(sentence):
        words_map[letter_idx] = word_idx
        if letter == ' ':
            word_idx += 1
    words_map[letter_idx + 1] = words_map[letter_idx]
    return words_map


if __name__ == '__main__':

    dataset_name: str
    for dataset_name in ['ca', 'eu']:
        convert_dataset()

    print(ERRORS)
