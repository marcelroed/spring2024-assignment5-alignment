from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from cs336_alignment.sft_dataset import PackedSFTDataset, iterate_batches
import deepspeed
from tqdm import tqdm
import wandb



def main():
    model_name = 'meta-llama/Meta-Llama-3-8B'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train_dataset = PackedSFTDataset(tokenizer=tokenizer, dataset_path='data/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz', seq_length=512, shuffle=True)
    
    print('Initializing distributed')
    deepspeed.init_distributed()
    print('Distributed initialized')

    print(deepspeed.dist.get_rank())

    gradient_accumulation_steps = 4

    for idx, batch in enumerate(tqdm(iterate_batches(train_dataset, batch_size=4, shuffle=True))):
        input_ids = batch['input_ids']
        labels = batch['labels']

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()

        if (idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()



if __name__ == '__main__':
    main()