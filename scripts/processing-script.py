from datasets import load_dataset
from transformers import AutoTokenizer
import glob
import os
import json
import pandas as pd
from datasets import Dataset, concatenate_datasets, ClassLabel


# tokenizer used in preprocessing
tokenizer_name = 'distilbert-base-uncased'

# dataset used
dataset_name = 'imdb'

# s3 key prefix for the data
s3_prefix = 'samples/datasets/imdb'


base_dir = "/opt/ml/processing"
base_output_dir = "/opt/ml/processing/output/"



# download tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# tokenizer helper function
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)


def augment_with_human_in_the_loop(initial_dataset):
    json_files = glob.glob(f"{base_dir}/input/data/**/*.json", recursive=True)
    df = pd.DataFrame()
    for file in json_files:
        try:
            print(f"---- processing file {file} -----")
            f = open(file)
            data = json.load(f)
            label = 'pos' if data['humanAnswers'][0]['answerContent']['category']['labels'][0] == "Positive" else 'neg'
            text = data['inputContent']['taskObject']

            df = df.append({'text': text, 'label':label}, ignore_index=True)
        except Exception as e:
            print(e)
    
    if df.empty:
        print("No json data found")
        return initial_dataset
    
    print("finished extraction from Json")
    
    ddf = Dataset.from_pandas(df)

    new_features = ddf.features.copy()
    new_features["label"] = ClassLabel(names=["neg", "pos"])
    ddf = ddf.cast(new_features)

    new_ddf = concatenate_datasets([ddf, initial_dataset])
    return new_ddf


if __name__ == "__main__":

    # load dataset
    train_dataset, test_dataset = load_dataset(dataset_name, split=['train', 'test'])
    train_dataset = train_dataset.shuffle().select(range(5000)) # smaller the size for test dataset to 5k to achieve lower training time for the purposes of the workshop 
    test_dataset = test_dataset.shuffle().select(range(1000)) # smaller the size for test dataset to 1k to achieve lower training time for the purposes of the workshop 

    train_dataset = augment_with_human_in_the_loop(train_dataset)
    
    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # set format for pytorch
    train_dataset =  train_dataset.rename_column("label", "labels")
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # save output
    train_dataset.save_to_disk(f'{base_output_dir}/train')
    test_dataset.save_to_disk(f'{base_output_dir}/test')    
    