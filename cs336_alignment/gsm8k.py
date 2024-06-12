from typing import Any, Literal
from pathlib import Path
from collections import defaultdict
import re
from pprint import pprint
from tqdm import tqdm
from cs336_alignment.llama import get_llama8b_multi, greedy_sampling_params

import pandas as pd


# Get the last occurrence of a number in the string.
NUMBER_RE = re.compile(r'\d+')

def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    all_numbers = NUMBER_RE.findall(model_output)

    if all_numbers:
        return all_numbers[-1]

    return None


class GSM8K:
    def __init__(self, base_path: Path | str = 'data/gsm8k'):
        if isinstance(base_path, str):
            base_path = Path(base_path)
        self.base_path = base_path
        self.data: dict[str, pd.DataFrame] = {}

        for split in ['test', 'train']:
            split_path = base_path / f'{split}.jsonl'
            with open(split_path, 'r') as f:
                split_df = pd.read_json(f, lines=True)
                self.data[split] = split_df
        
        with open('cs336_alignment/prompts/gsm8k.prompt', 'r') as f:
            self.prompt_template = f.read()
        
    def get_split(self, split: Literal['test', 'train'] = 'test') -> pd.DataFrame:
        return self.data[split]
    
    def format_prompt(self, prompt: dict[str, Any]):
        return self.prompt_template.format(**prompt)
    
    def evaluate_llm(self, llm_closure):
        result_df = pd.DataFrame()
        all_rows = [row for i, row in self.get_split('test').iterrows()]
        all_prompts = [self.format_prompt(dict(row)) for row in all_rows]

        result_df['prompt'] = all_prompts
        result_df['answer'] = [parse_gsm8k_response(row['answer']) for row in all_rows]

        pprint(all_prompts[:5])
        pd.set_option('display.max_columns', 4000)
        pd.set_option('display.max_colwidth', 4000)
        pd.set_option('display.width', 380)

        responses = llm_closure(all_prompts)
        result_df['response'] = responses
        pprint(responses[:5])
        parsed_responses = [parse_gsm8k_response(response) for row, response in zip(tqdm(all_rows), responses)]
        result_df['parsed_response'] = parsed_responses
        pprint(parsed_responses[:5])
        is_correct_response = [parsed == correct for parsed, correct in zip(parsed_responses, result_df['answer'])]
        result_df['correct'] = is_correct_response
        is_valid_response = [parsed is not None for parsed in parsed_responses]
        print([int(r) for r in is_correct_response][:5])
        print(f'Invalid responses: {sum(not valid for valid in is_valid_response)}/{len(is_valid_response)}')
        print(f'Accuracy: {sum(is_correct_response) / len(is_correct_response)}')
        print()
        print(result_df[result_df['parsed_response'].isnull()])

        incorrect_samples = result_df[result_df['correct'] == False].sample(10)
        for idx, row in incorrect_samples.iterrows():
            prompt = row['prompt'].replace('\n', r'\ ').replace('$', r'\$').replace('_', r'\_')
            print(f'[${idx}$], [{prompt}], [{row["response"].strip()}], [${row["parsed_response"]}$], [${row["answer"]}$],')


def main():
    gsm8k = GSM8K()
    print(gsm8k.data['test'].head())
    llm = get_llama8b_multi(num_gpus=1)

    llm_closure = lambda prompts: [output.outputs[0].text for output in llm.init_and_generate(prompts, greedy_sampling_params)]

    gsm8k.evaluate_llm(llm_closure)



if __name__ == '__main__':
    main()