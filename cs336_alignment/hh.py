import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import re
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import torch


class HHDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, seq_length: int, shuffle = True, dataset_path: str = 'data/hh', split: str = 'train', world_size=1, rank=0):
        root_dir = Path(dataset_path)
        pretokenized_path = root_dir / f'{split}_tokenized.jsonl'
        if pretokenized_path.exists():
            df = pd.read_json(pretokenized_path, lines=True)
        else:
            dfs = _get_hh_dataframes(root_dir=root_dir)
            df = dfs[split]
            
            # Tokenize the data
            with open('cs336_alignment/prompts/alpaca_sft.prompt', 'r') as f:
                prompt_template = f.read()
                prompt_template = f'{prompt_template}{tokenizer.eos_token}'
            
            full_prompts = [prompt_template.format(instruction=instruction, response=chosen_response) for instruction, chosen_response in zip(df['instruction'], df['chosen_response'])]
            tokenized_prompts = tokenizer(full_prompts).input_ids
            df['tokenized'] = tokenized_prompts
            print(df)
            df = df[['tokenized']]
            df.to_json(pretokenized_path, orient='records', lines=True)

        if shuffle:
            df = df.sample(frac=1, random_state=0).reset_index(drop=True)

        full_text_tok = torch.concatenate([torch.tensor(arr) for arr in df['tokenized']])

        chunk_size = len(full_text_tok) // world_size

        self.tokens = full_text_tok[rank * chunk_size:(rank + 1) * chunk_size]
        self.seq_length = seq_length
        self.shuffle = shuffle

    def __len__(self):
        return len(self.tokens) // self.seq_length
    
    def __getindex__(self, i: int):
        if i < 0 or i >= len(self):
            raise IndexError
        start_idx = i * self.seq_length
        end_idx = start_idx + self.seq_length
        return {
            'input_ids': self.tokens[start_idx:end_idx],
            'labels': self.tokens[start_idx+1:end_idx+1],
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
