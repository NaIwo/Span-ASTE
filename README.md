# Chunk-ASTE

## Brief description
This repository contains code for the Aspect Sentiment Triplet Extraction (ASTE) task. 
The solution (as of publication time) achieves some of the best results in the field. 
It reduces the complexity by working on automatically selected word phrases (chunk) and unifies the final results thanks to the CRF layer. 
Additionally, the selection layer supports the selection of correct phrases. 
Embedding of words are obtained thanks to BERT and embedding of phrases are aggregated using attention layer.

## Usage

#### Notes
- Tested on Python 3.8

#### Quick install
```
pip install -r requirements.txt
```

#### Dataset statistics
You can compute dataset statistics to examine your data.
One way to do this is to use our implementation. A full example is available in [dataset_statistics.py](./dataset_statistics.py)

#### Dataset preparation
```python
import os
from ASTE.dataset.reader import DatasetLoader

dataset_name: str = '14lap'
data_path: str = os.path.join(os.getcwd(), 'dataset', 'data', 'ASTE_data_v2', dataset_name)

# Create dataset reader. Path should point out directory with data - NOT EXACT DATA FILE
dataset_reader = DatasetLoader(data_path=data_path)

# Load data. Here, you should point to extract file with data.
train_data = dataset_reader.load('train.txt')
dev_data = dataset_reader.load('dev.txt')
test_data = dataset_reader.load('test.txt')
```
#### Experiments tracker
You can keep track of your experiments. We don't do that by default. 
If you want to track them, see the example in the file: [model_examples.py](./model_examples.py), 
where we discuss how to add 'experiments tracker'.

#### Setup trainer
```python
from ASTE.aste.trainer import Trainer
from ASTE.aste.models import BaseModel, BertBaseModel

# Get Model. You can create your own definition or update existing one.
# BertBase model is our provided implementation which obtain best results so far.
model: BaseModel = BertBaseModel()

# Define trainer - Wrapper for model handling.
trainer: Trainer = Trainer(model=model, tracker=tracker, save_path=save_path)
```
#### Model training
```python
trainer.train(train_data=train_data, dev_data=dev_data)
```

#### Model testing
```python
# LOAD BEST MODEL
trainer.load_model(save_path)

# You can check coverage of correct detected spans
trainer.check_coverage_detected_spans(test_data)

# Or you can test your model on selected dataset
results: Dict = trainer.test(test_data)
```

#### Saving the results
```python
from ASTE.aste.models import ModelMetric

results: Dict = trainer.test(test_data)
# You can also save the results. Just type:
save_path_json: str = os.path.join(os.getcwd(), 'results', 'name', 'results.json')
results[ModelMetric.NAME].to_json(path=save_path_json)
```

#### Querying the model
```python
from ASTE.aste.models import ModelOutput
from ASTE.dataset.domain.sentence import Sentence

# If you want to feed the model with your own sentence. 
# You should include the correct tokenization of the sentence (we are not handling that at the moment).
# You should create this sentence, and call predict:
sentence = Sentence('This OS is so fast !!')
prediction: ModelOutput = trainer.predict(sentence)
# You can also save the results to file
prediction.save('sentence_result.txt')
```

You can find usage examples in the file [model_examples.py](./model_examples.py)

### Configuration
You can adapt model configuration for your sepecific task in [config](./config.yml) file.

## Results
Results obtained on [ASTE_V2](https://aclanthology.org/2020.emnlp-main.183.pdf) 
and [MultiBooked](https://aclanthology.org/L18-1104.pdf) datasets. The latter was adapted for the ASTE task. 
It contains only two sentiment classes (POS, NEG). It can be found in the [this](./aste/dataset/multib) directory.

During the process of adapting the collection to the ASTE task, 
there were situations where the target or opinion was missing from a given extracted example - 
such an example was discarded (note that we are talking about an example from an aspect-opinion pair, 
not a whole sentence). During this process, we rejected 4 and 7 examples for the corresponding sets 'ca' and 'eu'.

|DatasetName|TripletPrecision|TripletRecall|TripletF1|
|-----------|----------------|-------------|---------|
|14lap      |73.68           |74.91        |74.24    |
|14res      |78.95           |81.93        |80.36    |
|15res      |67.31           |72.02        |69.58    |
|16res      |74.57           |75.99        |75.24    |
|ca         |65.79           |64.97        |64.88    |
|eu         |66.03           |66.83        |66.22    |

## Solution architecture
![Architecture](./assets/chunk-ASTE.png)