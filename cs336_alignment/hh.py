import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import re
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import torch
from cs336_alignment.dpo import per_instance_dpo_loss


class HHDataset(Dataset):
    def __init__(self, *, shuffle = True, dataset_path: str = 'data/hh', split: str = 'train', get_val=True):
        root_dir = Path(dataset_path)

        dfs = _get_hh_dataframes(root_dir=root_dir)
        df = dfs[split]

        if shuffle:
            df = df.sample(frac=1, random_state=0).reset_index(drop=True)

        # full_text_tok = torch.concatenate([torch.tensor(arr) for arr in df['tokenized']])

        # chunk_size = len(full_text_tok) // world_size

        # self.tokens = full_text_tok[rank * chunk_size:(rank + 1) * chunk_size]
        # print(f'full: {df.head()=}')
        if get_val:
            self.data = df.iloc[-200:]
        else:
            self.data = df.iloc[:-200]
        # print(f'{self.data.head()=}')
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i: int):
        if i < 0 or i >= len(self):
            raise IndexError

        row = self.data.iloc[i]

        # print(f'{row=}')

        return {
            'prompt': row['instruction'],
            'response_chosen': row['chosen_response'],
            'response_rejected': row['rejected_response'],
        }



def _read_hh_file(dataset_path: Path):
    name = dataset_path.parent.name
    split_name = dataset_path.name

    df = pd.read_json(dataset_path, lines=True)

    chosen_texts = df['chosen']
    start_count = len(chosen_texts)
    single_response = chosen_texts.str.count('Assistant:') == 1

    df = df[single_response].reset_index(drop=True)

    df['instruction'] = df['chosen'].str.extract(r'Human: ([\s\S]*)Assistant:', flags=re.MULTILINE).squeeze()
    df['chosen_response'] = df['chosen'].str.extract(r'Assistant: ([\s\S]*)', flags=re.MULTILINE).squeeze()
    df['rejected_response'] = df['rejected'].str.extract(r'Assistant: ([\s\S]*)', flags=re.MULTILINE).squeeze()

    print(f'{name}({split_name}): {len(df)}/{start_count} samples')

    df.drop(columns=['chosen', 'rejected'], inplace=True)
    df['dataset_name'] = name

    return df


def _get_hh_dataframes(root_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for split in 'train', 'test':
        dfs = []
        for dataset_path in tqdm(list(root_dir.rglob(f'{split}.jsonl.gz')), desc='Loading HH datasets'):
            df = _read_hh_file(dataset_path)
            dfs.append(df)

        out[split] = pd.concat(dfs, ignore_index=True)

    return out


if __name__ == '__main__':
    dataset = HHDataset(tokenizer=AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B'), seq_length=512, shuffle=True, dataset_path='data/hh', split='test')
    print(dataset.tokens.shape)
