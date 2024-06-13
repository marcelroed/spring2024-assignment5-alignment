from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from cs336_alignment.dpo import per_instance_dpo_loss
from cs336_alignment.hh import HHDataset
from cs336_alignment.sft_dataset import PackedSFTDataset, iterate_batches
import deepspeed
import shutil
from itertools import islice
from tqdm import tqdm, trange
import wandb
import os
from pathlib import Path
from cs336_alignment.checkpoints import load_deepspeed_checkpoint



def main():
    n_epochs = 1
    # Needs to be run with deepspeed command
    model_name = 'meta-llama/Meta-Llama-3-8B'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_ref = load_deepspeed_checkpoint(device_map='cuda:1')
    for param in model_ref.parameters():
        param.requires_grad_(False)

    model_pi = load_deepspeed_checkpoint(device_map='cuda:0')
    print('Finished loading model to CPU')

    train_dataset = HHDataset(dataset_path='data/hh', shuffle=True, split='train', get_val=False)
    print('Loaded dataset')

    microbatch_size = 1
    total_num_steps = len(train_dataset) * n_epochs // microbatch_size
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(model=model_pi, model_parameters=model_pi.parameters(), config={
        'train_batch_size': 64,
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
        # 'zero_optimization': {
        #     'stage': 2,
        #     'overlap_comm': True,
        #     # 'zero_quantized_weights': True,
        #     # 'zero_quantized_gradients': True,
        # },
        'bf16': {
            'enabled': True,
        },
        'gradient_clipping': 1.0,
    })
    model_engine: deepspeed.DeepSpeedEngine

    logging_rate = 10
    valid_rate = 1000 

    wandb.init(project='cs336-alignment', entity='marcelroed', name='dpo_train', group='llama8b', config={
        'model_name': model_name,
        'batch_size': 64,
        'seq_length': 512,
        'learning_rate': 1e-6,
        'total_num_steps': total_num_steps,
    })

    epoch_range = trange(n_epochs, desc='Epochs')

    train_dataloader = train_dataset
    
    valid_dataset = HHDataset(dataset_path='data/hh', shuffle=True, split='train', get_val=True)

    valid_dataloader = valid_dataset

    step = 0

    best_val_loss = float('inf')

    best_chkpnt_path = Path('dpo_model_llama3_8b_best.ckpt')

    def get_loss(lm, lm_ref, prompt: str, response_rejected: str, response_chosen: str):
        return per_instance_dpo_loss(lm=lm, lm_ref=lm_ref, tokenizer=tokenizer, beta=0.1, prompt=prompt, response_chosen=response_chosen, response_rejected=response_rejected)

    for epoch in epoch_range:
        pbar = enumerate(tqdm(train_dataloader, desc='Training'))
        for idx, batch in pbar:
            loss = get_loss(model_engine, model_ref, **batch)

            model_engine.backward(loss)

            model_engine.step()
            log_dict = {}
            if idx % valid_rate == 0:
                log_dict.update(**{'loss': loss.item(), 'learning_rate': optimizer.param_groups[0]['lr']})
                del loss
                loss_values = []
                val_pbar = tqdm(valid_dataloader, desc='Computing validation loss')
                with torch.no_grad():
                    for val_idx, val_batch in enumerate(val_pbar):
                        # val_input_ids = val_batch['input_ids'].cuda()
                        # val_labels = val_batch['labels'].cuda()

                        val_loss = get_loss(model_engine, model_ref, **val_batch)
                        loss_values.append(val_loss)


                    val_loss = torch.mean(torch.tensor(loss_values))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if best_chkpnt_path.exists():
                            shutil.rmtree(best_chkpnt_path)
                        model_engine.save_checkpoint(str(best_chkpnt_path))
                    log_dict.update(**{'valid_loss': torch.mean(torch.tensor(loss_values))})
                    wandb.log(log_dict)
                    del loss_values
            elif idx % logging_rate == 0:
                log_dict.update(**{'loss': loss.item(), 'learning_rate': optimizer.param_groups[0]['lr']})
                del loss
                wandb.log(log_dict, step=step)
            step += 1

        print(f'Finished epoch {epoch} after {idx} microbatches')
    # model_engine.save_checkpoint(f'sft_model_llama3_8b_unshifted_final.ckpt')



if __name__ == '__main__':
    main()