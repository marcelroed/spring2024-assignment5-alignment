import torch
from transformers import PreTrainedTokenizerBase
from math import log
import torch.nn.functional as F

def per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
):
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    with open('cs336_alignment/prompts/alpaca_sft.prompt', 'r') as f:
        prompt_template = f.read()
        prompt_template = f'{prompt_template}{tokenizer.eos_token}'

    chosen_prompt = prompt_template.format(instruction=prompt, response=response_chosen)
    rejected_prompt = prompt_template.format(instruction=prompt, response=response_rejected)

    chosen_tokens, rejected_tokens = tokenizer([chosen_prompt, rejected_prompt], return_tensors='pt').input_ids

    chosen_ll = -lm(input_ids=chosen_tokens[:-1], labels=chosen_tokens[1:]).loss * len(chosen_tokens)
    rejected_ll = -lm(input_ids=rejected_tokens[:-1], labels=rejected_tokens[1:]).loss * len(rejected_tokens)

    chosen_ref_ll = -lm_ref(input_ids=chosen_tokens[:-1], labels=chosen_tokens[1:]).loss * len(chosen_tokens)
    reject_ref_ll = -lm_ref(input_ids=rejected_tokens[:-1], labels=rejected_tokens[1:]).loss * len(rejected_tokens)

    log_core = chosen_ll - chosen_ref_ll - rejected_ll + reject_ref_ll

    return -F.logsigmoid(beta * log_core)