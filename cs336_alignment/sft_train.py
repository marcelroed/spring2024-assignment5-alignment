from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from cs336_alignment.sft_dataset import PackedSFTDataset, iterate_batches
import deepspeed
from tqdm import tqdm, trange
import wandb
import os



def main():
    n_epochs = 10
    # Needs to be run with deepspeed command
    model_name = 'meta-llama/Meta-Llama-3-8B'
    rank = int(os.environ['RANK'])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )
    print('Finished loading model to CPU')


    train_dataset = PackedSFTDataset(tokenizer=tokenizer, dataset_path='data/safety_augmented_ultrachat_200k_single_turn/train.jsonl', seq_length=512, shuffle=True, world_size=8, rank=rank)
    print('Loaded dataset')
    total_num_steps = len(train_dataset) * n_epochs

    microbatch_size = 2
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, model_parameters=model.parameters(), config={
        'train_batch_size': 32,
        'train_micro_batch_size_per_gpu': microbatch_size,
        # 'optimizer': {
        #     'FusedAdam': {
        #         'params': {
        #             'lr': 2e-5,
        #         },
        #         'adam_w_mode': True,
        #     },
        # },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 2e-5,
            },
        },
        'scheduler': {
            'type': 'WarmupCosineLR',
            "params": {
                'warmup_min_ratio': 0,
                'warmup_num_steps': int(total_num_steps * 0.03),
                'total_num_steps': total_num_steps, 
                'warmup_type': 'linear',
            }
        },
        'zero_optimization': {
            'stage': 2,
            'overlap_comm': True,
            # 'zero_quantized_weights': True,
            # 'zero_quantized_gradients': True,
        },
        'bf16': {
            'enabled': True,
        },
        'gradient_clipping': 1.0,


    })
    model_engine: deepspeed.DeepSpeedEngine
    del model, tokenizer

    print(f'Rank: {rank}')

    logging_rate = 1
    rank = deepspeed.dist.get_rank()

    if rank == 0:
        wandb.init(project='cs336-alignment', entity='marcelroed', name='sft_train', group='llama8b', config={
            'model_name': model_name,
            'batch_size': 32,
            'seq_length': 512,
            'learning_rate': 2e-5,
            'total_num_steps': total_num_steps,
        })

    for epoch in trange(n_epochs):
        print(f'Rank {rank} has {len(train_dataset)} examples')
        for idx, batch in enumerate(tqdm(iterate_batches(train_dataset, batch_size=microbatch_size, shuffle=True))):
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()

            outputs = model_engine(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            if rank == 0 and (idx % logging_rate == 0):
                wandb.log({'loss': loss.item(), 'learning_rate': optimizer.param_groups[0]['lr']}, step=idx + epoch * len(train_dataset) // microbatch_size)

            model_engine.backward(loss)
            del loss
            del outputs
            del labels
            del input_ids

            model_engine.step()
        print(f'Rank {rank} finished epoch {epoch} after {idx} microbatches')
    model_engine.save_checkpoint(f'sft_model_llama3_8b_final.ckpt')



if __name__ == '__main__':
    main()