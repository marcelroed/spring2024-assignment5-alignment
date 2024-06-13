import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import re


def get_hh_dataset():
    out = {}
    for dataset_path in tqdm(list(Path('data/hh').rglob('train.jsonl.gz')), desc='Loading HH datasets'):
        name = dataset_path.parent.name
        df = pd.read_json(dataset_path, lines=True)

        chosen_texts = df['chosen']
        start_count = len(chosen_texts)
        single_response = chosen_texts.str.count('Assistant:') == 1

        df = df[single_response].reset_index(drop=True)

        df['instruction'] = df['chosen'].str.extract(r'Human: ([\s\S]*)Assistant:', flags=re.MULTILINE).squeeze()
        df['chosen_response'] = df['chosen'].str.extract(r'Assistant: ([\s\S]*)', flags=re.MULTILINE).squeeze()
        df['rejected_response'] = df['rejected'].str.extract(r'Assistant: ([\s\S]*)', flags=re.MULTILINE).squeeze()

        out[name] = df
        print(f'{name}: {len(df)}/{start_count} samples')

        df.drop(columns=['chosen', 'rejected'], inplace=True)

    return out


if __name__ == '__main__':
    d = get_hh_dataset()
    print(d)
