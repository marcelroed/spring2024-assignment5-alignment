from transformers import PreTrainedTokenizerBase, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

class PackedSFTDataset(Dataset):
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, dataset_path: str, seq_length: int, shuffle: bool, world_size=1, rank=0):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path

        if dataset_path.endswith('safety_augmented_ultrachat_200k_single_turn/train.jsonl'):
            df = pd.read_json('data/safety_augmented_ultrachat_200k_single_turn/train_tokenized.jsonl', lines=True)
        else:  # Pretokenized in this case
            df = pd.read_json(dataset_path, dtype={'prompt': str, 'response': str}, lines=True)
            print('Finished reading JSON')

            prompts, responses = df['prompt'], df['response']

            with open('cs336_alignment/prompts/alpaca_sft.prompt', 'r') as f:
                prompt_template = f.read()
                prompt_template = f'{prompt_template}{tokenizer.eos_token}'

            full_prompts = [prompt_template.format(instruction=prompt, response=response) for prompt, response in zip(prompts, responses)]
            tokenized_prompts = tokenizer(full_prompts).input_ids
            df['tokenized'] = tokenized_prompts
            df = df[['tokenized']]

        if shuffle:
            df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        
        full_text_tok = torch.concatenate([torch.tensor(arr) for arr in df['tokenized']])


        # interleaved = [torch.tensor(elem) for toks in tokenized_prompts for elem in (toks, [tokenizer.bos_token_id])][:-1]

        # full_text = tokenizer.bos_token.join(full_prompts)
        # print(full_text)

        # full_text_tok = self.tokenizer(full_text).input_ids
        # print(interleaved)
        print('Tokenized prompts')
        # print(type(full_text_tok))

        chunk_size = len(full_text_tok) // world_size

        self.tokens = full_text_tok[rank * chunk_size:(rank + 1) * chunk_size]
        # print(self.tokens.shape)

        self.seq_length = seq_length
        self.shuffle = shuffle
    
    def __len__(self):
        return len(self.tokens) // self.seq_length

    def __getitem__(self, i: int):
        if i < 0 or i >= len(self):
            raise IndexError
        start_idx = i * self.seq_length
        end_idx = start_idx + self.seq_length
        return {
            'input_ids': self.tokens[start_idx:end_idx],
            'labels': self.tokens[start_idx+1:end_idx+1]
        }
        

def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    print(tokenizer.decode([13]))
    print(tokenizer.decode([627]))
    print(tokenizer.decode([128000]))
    print(tokenizer.decode([128001]))
    ds = PackedSFTDataset(tokenizer, 'data/safety_augmented_ultrachat_200k_single_turn/small_train.jsonl', 128, True)
