from transformers import PreTrainedTokenizerBase, AutoTokenizer
from torch.utils.data import Dataset
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
    def __init__(self, tokenizer: PreTrainedTokenizerBase, dataset_path: str, seq_length: int, shuffle: bool):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path

        df = pd.read_json(dataset_path, dtype={'prompt': str, 'response': str}, lines=True)

        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        
        print(df)

        prompts, responses = df['prompt'], df['response']

        prompts_tok = self.tokenizer(prompts.tolist())
        responses_tok = self.tokenizer(responses.tolist())

        self.prompts_tok = torch.tensor(prompts_tok.input_ids)
        self.responses_tok = torch.tensor(responses_tok.input_ids)

        self.seq_length = seq_length
        self.shuffle = shuffle
    
    def __len__(self):
        return len(self.prompts_tok)

    def __getitem__(self, i):
        return {
            'input_ids': self.prompts_tok[i][:self.seq_length],
            'labels': self.responses_tok[i][:self.seq_length]
        }
        

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    ds = PackedSFTDataset(tokenizer, 'data/safety_augmented_ultrachat_200k_single_turn/small_train.jsonl', 128, True)
