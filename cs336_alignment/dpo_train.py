from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from cs336_alignment.dpo import per_instance_dpo_loss, prefers_correct
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
    # for param in model_ref.parameters():
    #     param.requires_grad_(False)

    model_pi = load_deepspeed_checkpoint(device_map='cuda:0')
    print('Finished loading model to CPU')
    # for param in model_pi.parameters():
    #     print(f'{param.requires_grad=}')
    #     param.requires_grad_(True)

    train_dataset = HHDataset(dataset_path='data/hh', shuffle=True, split='train', get_val=False)
    print('Loaded dataset')

    microbatch_size = 1
    num_microbatches = 64
    total_num_steps = len(train_dataset) * n_epochs // num_microbatches

    print('constructing optimizer')
    optimizer = torch.optim.RMSprop(model_pi.parameters(), lr=1e-6)
    print('deepspeed init')
    # input()
    # _, optimizer, _, lr_scheduler = deepspeed.initialize(optimizer=optimizer, model_parameters=model_pi.parameters(), config={
    #     'train_batch_size': 64,
    #     'train_micro_batch_size_per_gpu': microbatch_size,
    #     # 'optimizer': {
    #     #     'FusedAdam': {
    #     #         'params': {
    #     #             'lr': 2e-5,
    #     #         },
    #     #         'adam_w_mode': True,
    #     #     },
    #     # },
    #     # "optimizer": {
    #     #     "type": "RMSProp",
    #     #     "params": {
    #     #         "lr": 1e-6,
    #     #     },
    #     # },
    #     'scheduler': {
    #         'type': 'WarmupCosineLR',
    #         "params": {
    #             'warmup_min_ratio': 0,
    #             'warmup_num_steps': int(total_num_steps * 0.03),
    #             'total_num_steps': total_num_steps, 
    #             'warmup_type': 'linear',
    #         }
    #     },
    #     # 'zero_optimization': {
    #     #     'stage': 2,
    #     #     'overlap_comm': True,
    #     #     # 'zero_quantized_weights': True,
    #     #     # 'zero_quantized_gradients': True,
    #     # },
    #     # 'bf16': {
    #     #     'enabled': True,
    #     # },
    #     'gradient_clipping': 1.0,
    # })
    # print('deepspeed init done')
    # input()
    # model_engine: deepspeed.DeepSpeedEngine

    # del model_pi


    logging_rate = 10
    valid_rate = 1000

    wandb.init(project='cs336-alignment', entity='marcelroed', name='dpo_train', group='llama8b', config={
        'model_name': model_name,
        'batch_size': 64,
        'learning_rate': 1e-6,
        'total_num_steps': total_num_steps,
    })

    epoch_range = trange(n_epochs, desc='Epochs')

    train_dataloader = train_dataset
    
    valid_dataset = HHDataset(dataset_path='data/hh', shuffle=True, split='train', get_val=True)

    valid_dataloader = valid_dataset

    step = 0

    best_val_acc = 0.0

    best_chkpnt_path = Path('dpo_model_llama3_8b_best.ckpt')

    def get_loss(lm, lm_ref, prompt: str, response_rejected: str, response_chosen: str):
        return 
    
    # Show devices for each model
    # print(next(model_engine.parameters()).device)
    # print(next(model_ref.parameters()).device)
    print(f'{len(train_dataloader)=}')

    for epoch in epoch_range:
        pbar = enumerate(tqdm(train_dataloader, desc='Training'))
        train_losses = []
        for idx, batch in pbar:
            loss = per_instance_dpo_loss(lm=model_pi, lm_ref=model_ref, tokenizer=tokenizer, beta=0.1, prompt=batch['prompt'], response_chosen=batch['response_chosen'], response_rejected=batch['response_rejected'])

            loss.backward()
            train_losses.append(loss.detach().item())

            if (idx + 1) % num_microbatches == 0:
                # print('TAKING OPTIMIZER STEP')
                # # Show that optimizer states are non-zero
                # for k, v in model_pi.state_dict().items():
                #     if hasattr(v, 'grad'):
                #         print(k, v.grad, v.requires_grad)
                # for v in model_pi.parameters():
                #     if hasattr(v, 'grad'):
                #         print(v.grad, v.requires_grad)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
            log_dict = {}
            if idx % valid_rate == 0:
                log_dict.update(**{'loss': loss.detach().item(), 'learning_rate': optimizer.param_groups[0]['lr']})
                del loss
                correct_pred = []
                val_pbar = tqdm(valid_dataloader, desc='Computing validation loss')
                with torch.no_grad():
                    for val_idx, val_batch in enumerate(val_pbar):
                        # val_input_ids = val_batch['input_ids'].cuda()
                        # val_labels = val_batch['labels'].cuda()

                        val_correct = prefers_correct(model_pi, tokenizer, **val_batch)
                        correct_pred.append(val_correct)


                    val_acc = torch.mean(torch.tensor(correct_pred).float()).item()
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        if best_chkpnt_path.exists():
                            best_chkpnt_path.unlink()
                        torch.save(model_pi.state_dict(), best_chkpnt_path)
                        # model_engine.save_checkpoint(str(best_chkpnt_path))
                    log_dict.update(**{'val_acc': val_acc})
                    wandb.log(log_dict)
                    del correct_pred
            elif idx % logging_rate == 0:
                train_loss = torch.mean(torch.tensor(train_losses)).item()
                log_dict.update(**{'train_loss': train_loss, 'learning_rate': optimizer.param_groups[0]['lr']})
                train_losses = []
                wandb.log(log_dict, step=idx)

        print(f'Finished epoch {epoch} after {idx} microbatches')
    # model_engine.save_checkpoint(f'sft_model_llama3_8b_unshifted_final.ckpt')


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])

if __name__ == '__main__':
    main()