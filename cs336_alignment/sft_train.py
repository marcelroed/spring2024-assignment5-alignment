from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from cs336_alignment.sft_dataset import PackedSFTDataset, iterate_batches
import deepspeed
from itertools import islice
from tqdm import tqdm, trange
import wandb
import os



def main():
    n_epochs = 5
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

    microbatch_size = 4
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
    del model

    print(f'Rank: {rank}')

    logging_rate = 10
    valid_rate = 1000 
    rank = deepspeed.dist.get_rank()

    if rank == 0:
        wandb.init(project='cs336-alignment', entity='marcelroed', name='sft_train', group='llama8b', config={
            'model_name': model_name,
            'batch_size': 32,
            'seq_length': 512,
            'learning_rate': 2e-5,
            'total_num_steps': total_num_steps,
        })

    if rank == 0:
        epoch_range = trange(n_epochs, desc='Epochs')
    else:
        epoch_range = range(n_epochs)

    train_dataloader = iterate_batches(train_dataset, batch_size=microbatch_size, shuffle=True)
    
    valid_dataset = PackedSFTDataset(tokenizer=tokenizer, dataset_path='data/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz', seq_length=512, shuffle=False, world_size=8, rank=rank)

    valid_dataloader = iterate_batches(valid_dataset, batch_size=microbatch_size, shuffle=False)

    step = 0

    for epoch in epoch_range:
        if rank == 0:
            pbar = enumerate(tqdm(train_dataloader, desc='Training'))
        else:
            pbar = enumerate(train_dataloader)
        for idx, batch in pbar:
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()

            outputs = model_engine(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            model_engine.backward(loss)
            del outputs
            del labels
            del input_ids

            model_engine.step()
            log_dict = {}
            if idx % valid_rate == 0:
                log_dict.update(**{'loss': loss.item(), 'learning_rate': optimizer.param_groups[0]['lr']})
                del loss
                loss_values = []
                if rank == 0:
                    val_pbar = tqdm(valid_dataloader, desc='Computing validation loss')
                else:
                    val_pbar = valid_dataloader

                for val_idx, val_batch in enumerate(val_pbar):
                    val_input_ids = val_batch['input_ids'].cuda()
                    val_labels = val_batch['labels'].cuda()

                    val_outputs = model_engine(input_ids=val_input_ids, labels=val_labels)
                    val_loss = val_outputs.loss.detach()

                    deepspeed.dist.all_reduce(val_loss, op=deepspeed.dist.ReduceOp.AVG)
                    loss_values.append(val_loss)

                    del val_outputs
                    del val_labels
                    del val_input_ids
                if rank == 0:
                    log_dict.update(**{'valid_loss': torch.mean(torch.tensor(loss_values))})
                    wandb.log(log_dict)
                del loss_values
            elif rank == 0 and idx % logging_rate == 0:
                log_dict.update(**{'loss': loss.item(), 'learning_rate': optimizer.param_groups[0]['lr']})
                del loss
                wandb.log(log_dict, step=step)
            step += 1

        print(f'Rank {rank} finished epoch {epoch} after {idx} microbatches')
    model_engine.save_checkpoint(f'sft_model_llama3_8b_final.ckpt')



if __name__ == '__main__':
    main()