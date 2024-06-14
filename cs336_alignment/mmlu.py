from typing import Any, Literal
from pathlib import Path
from collections import defaultdict
import pandas as pd
import re
from tqdm import tqdm
from pprint import pprint

from cs336_alignment.llama import get_llama8b_dpo_multi

# Single capitalized letter from A to D (can be followed with any punctiation or whitespace, just no other letters on either side)
ANSWER_RE = re.compile(r'(?<![A-z])[A-D](?![A-z])')

def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str):
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    res = ANSWER_RE.search(model_output)

    if res:
        return res.group()
    
    return None


class MMLU:
    def __init__(self, base_path: Path | str):
        if isinstance(base_path, str):
            base_path = Path(base_path)
        self.base_path = base_path
        self.data = defaultdict(lambda: defaultdict(list))

        for split in ['dev', 'test', 'val']:
            split_dir = base_path / split
            print(split_dir)
            for subject_path in split_dir.iterdir():
                subject_id = subject_path.stem
                subject_id = subject_id[:subject_id.rfind('_')]
                df = pd.read_csv(subject_path, header=None, names=['question', 'o1', 'o2', 'o3', 'o4', 'answer'])
                records = df.to_dict('records')
                for row in records:
                    row['options'] = [row.pop(f'o{i}') for i in range(1, 5)]
                    row['subject'] = ' '.join(subject_id.split('_'))
                self.data[split][subject_id] = records
        
        with open('cs336_alignment/prompts/mmlu.prompt', 'r') as f:
            self.prompt_template = f.read()

    def get_keys(self):
        return self.data['test'].keys()
    
    def get_subject(self, key: str, split: Literal['dev', 'test', 'val'] = 'test'):
        return self.data[split][key]
    
    def format_prompt(self, prompt: dict[str, Any]):
        return self.prompt_template.format(**prompt)
    
    def get_subject_formatted(self, key: str, split: Literal['dev', 'test', 'val'] = 'test'):
        subject = self.get_subject(key, split)
        return [self.format_prompt(p) for p in subject]
    
    def evaluate_llm(self, llm_closure):
        result_df = pd.DataFrame()
        all_rows = [row for key in self.get_keys() for row in self.get_subject(key)]
        all_prompts = [self.format_prompt(row) for row in all_rows]

        result_df['prompt'] = all_prompts
        result_df['answer'] = [row['answer'] for row in all_rows]

        pprint(all_prompts[:5])
        pd.set_option('display.max_columns', 4000)
        pd.set_option('display.max_colwidth', 4000)
        pd.set_option('display.width', 380)

        responses = llm_closure(all_prompts)
        result_df['response'] = responses
        pprint(responses[:5])
        parsed_responses = [parse_mmlu_response(row, response) for row, response in zip(tqdm(all_rows), responses)]
        result_df['parsed_response'] = parsed_responses
        pprint(parsed_responses[:5])
        correct_responses = [row['answer'] for row in all_rows]
        is_correct_response = [parsed == correct for parsed, correct in zip(parsed_responses, correct_responses)]
        result_df['correct'] = is_correct_response
        is_valid_response = [parsed is not None for parsed in parsed_responses]
        print([int(r) for r in is_correct_response][:5])
        print(f'Invalid responses: {sum(not valid for valid in is_valid_response)}/{len(is_valid_response)}')
        print(f'Accuracy: {sum(is_correct_response) / len(is_correct_response)}')
        print()
        print(result_df[result_df['parsed_response'].isnull()].iloc[:5])

        incorrect_samples = result_df[result_df['correct'] == False].sample(10)
        for idx, row in incorrect_samples.iloc[:5].iterrows():
            prompt = row['prompt'].replace('\n', r'\ ').replace('$', r'\$').replace('_', r'\_')
            print(f'[${idx}$], [{prompt}], [{row["response"].strip()}], [{row["parsed_response"]}], [{row["answer"]}],')




def run_llama_zero_shot():
    from cs336_alignment.llama import get_llama8b_multi, greedy_sampling_params, get_llama8b_sft_multi
    # mmlu = MMLU('data/mmlu/')
    # print(mmlu.get_keys())
    # print(mmlu.format_prompt(mmlu.get_subject('virology')[0]))
    # llm = get_llama8b_multi()
    # llm = get_llama8b_sft_multi()
    llm = get_llama8b_dpo_multi()
    llm_closure = lambda ss: [output.outputs[0].text for output in llm.init_and_generate(ss, greedy_sampling_params)]

    mmlu = MMLU('data/mmlu/')
    mmlu.evaluate_llm(llm_closure)



    


def main():
    run_llama_zero_shot()


if __name__ == '__main__':
    main()