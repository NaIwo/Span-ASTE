import json
from os.path import join
from collections import defaultdict
from typing import List, Dict, DefaultDict

SENTIMENT_MAP = {
    'Positive': 'POS',
    'Negative': 'NEG',
}

SEP: str = '#### #### ####'

ERRORS: DefaultDict = defaultdict(int)


def convert_dataset() -> None:
    for data_source in ['train.json', 'dev.json', 'test.json']:
        data_path: str = join('.', dataset_name, data_source)
        with open(data_path, 'r') as file:
            data: List = json.load(file)

        convert_sentences(data, data_path)


def convert_sentences(data: List, data_path: str) -> None:
    save_path: str = data_path.replace('json', 'txt')

    row: Dict
    for row in data:
        sentence = row['text'] + SEP

        triplets: List = get_triplets_from_row(row)

        if triplets:
            sentence += str(triplets)
            with open(save_path, 'a', encoding='utf-8') as file:
                file.write(sentence + '\n')


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

            triplets.append((aspect, opinion, SENTIMENT_MAP[sentiment]))

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
