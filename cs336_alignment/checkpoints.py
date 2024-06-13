from pathlib import Path
import torch
import deepspeed
from transformers import AutoModelForCausalLM


def load_deepspeed_checkpoint(checkpoint_path: Path = Path('/workspace/spring2024-assignment5-alignment/sft_model_llama3_8b_unshifted_best.ckpt/global_step13455/mp_rank_00_model_states.pt'), device_map='auto'):
    states = torch.load(checkpoint_path)['module']
    print(states)
    print(states.__dict__)
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map=device_map)
    model.load_state_dict(states)
    return model

if __name__ == '__main__':
    checkpoint_path = Path('/workspace/spring2024-assignment5-alignment/sft_model_llama3_8b_unshifted_best.ckpt/global_step13455/mp_rank_00_model_states.pt')

    load_deepspeed_checkpoint(checkpoint_path)