import os
import json
import torch
import logging
import argparse
import random
import shutil
import math
import wandb
import numpy as np
import gc
from typing import List, Dict

import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from einops import rearrange
from transformers import set_seed, AutoModelForCausalLM
from PIL import Image

from decord import VideoReader, cpu
import torchvision.transforms as transforms

from janus.models import VLChatProcessor, ActionTokenizer
from models.cosmos_janus import CosmosJanusMoT
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_custom_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
        return scaled_factor
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

class VLADataset(Dataset):
    def __init__(self, config, processor, accelerator):
        self.config = config
        self.processor = processor
        self.accelerator = accelerator
        self.tokenizer = processor.tokenizer
        
        self.action_tokenizer = ActionTokenizer(self.tokenizer, need_to_sub=3)
        
        self.accelerator.print(f"Loading dataset from {config.data_path} ...")
        with open(config.data_path, 'r') as f:
            self.data = json.load(f)

        statistics_path = config.data_path.replace(".json", "_statistics.json")
        with open(statistics_path, 'r') as f:
            self.stats_data = json.load(f)
            
        self.dataset_name = next(iter(self.stats_data))
        self.action_q01 = np.array(self.stats_data[self.dataset_name]['action']['q01'])
        self.action_q99 = np.array(self.stats_data[self.dataset_name]['action']['q99'])
        self.action_mask = np.array(self.stats_data[self.dataset_name]['action']['mask'])
        self.state_q01 = np.array(self.stats_data[self.dataset_name]['state']['q01'])
        self.state_q99 = np.array(self.stats_data[self.dataset_name]['state']['q99'])
        self.state_mask = np.array(self.stats_data[self.dataset_name]['state']['mask'])

        self.video_transform = transforms.Compose([
            transforms.Resize(min(config.video_h, config.video_w), antialias=True),
            transforms.CenterCrop((config.video_h, config.video_w)),
        ])

    def __len__(self):
        return len(self.data)
    def _normalize(self, data_array, q01, q99, mask):
        return np.where(
            mask,
            np.clip(2 * (data_array - q01) / (q99 - q01 + 1e-8) - 1.0, -1.0, 1.0),
            data_array
        )
    def _load_video(self, video_path, start_frame):
        target_frames = self.config.video_frames 
        
        if hasattr(self.config, 'data_root') and self.config.data_root and not os.path.isabs(video_path):
            video_path = os.path.join(self.config.data_root, video_path)

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        indices = np.arange(start_frame, start_frame + target_frames)
        indices = np.clip(indices, 0, total_frames - 1)
        
        frames = vr.get_batch(indices).asnumpy()

        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        frames_tensor = self.video_transform(frames_tensor)

        frames_tensor = frames_tensor.permute(1, 0, 2, 3)
        return frames_tensor

    def __getitem__(self, index):
        sample = self.data[index]
        
        state_tokens_str = ""
        if self.config.robot_state and 'state' in sample and self.state_mask is not None:
            state_arr = np.array(sample['state'], dtype=np.float32)
            norm_state = self._normalize(state_arr, self.state_q01, self.state_q99, self.state_mask)
            state_tokens_str = self.action_tokenizer(norm_state)

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{sample['input_prompt']}{state_tokens_str}",
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        frame_idx = sample.get('frame_index', 0)
        video_tensor = self._load_video(sample['video_path'], frame_idx)

        vr = VideoReader(sample['video_path'] if os.path.isabs(sample['video_path']) else os.path.join(self.config.data_root, sample['video_path']), ctx=cpu(0))
        first_frame_np = vr.get_batch([frame_idx]).asnumpy()[0]
        first_frame_pil = Image.fromarray(first_frame_np.astype(np.uint8))
        
        janus_inputs = self.processor(
            conversations=conversation, 
            images=[first_frame_pil], 
            return_tensors="pt"
        )

        action_arr = np.array(sample['action'], dtype=np.float32).reshape(-1, self.config.action_dim)
        if action_arr.shape[0] < self.config.action_chunk:
            pad_len = self.config.action_chunk - action_arr.shape[0]
            pad_actions = np.repeat(action_arr[-1:], pad_len, axis=0)
            action_arr = np.concatenate([action_arr, pad_actions], axis=0)
        elif action_arr.shape[0] > self.config.action_chunk:
            action_arr = action_arr[:self.config.action_chunk]
            
        norm_action = self._normalize(action_arr, self.action_q01, self.action_q99, self.action_mask)
        actions_tensor = torch.tensor(norm_action, dtype=torch.float32)

        return {
            "janus_input_ids": janus_inputs.input_ids.squeeze(0),           
            "janus_pixel_values": janus_inputs.pixel_values.squeeze(0), # [n, c, h, w]
            "janus_images_seq_mask": janus_inputs.images_seq_mask.squeeze(0),
            "janus_images_emb_mask": janus_inputs.images_emb_mask.squeeze(0),
            "actions": actions_tensor,        
            "videos": video_tensor            
        }
        
    def collate_fn(self, batch):
        input_ids_list = [x['janus_input_ids'] for x in batch]
        seq_mask_list = [x['janus_images_seq_mask'] for x in batch]
        max_len = max(len(ids) for ids in input_ids_list)
        
        padded_input_ids = []
        padded_seq_masks = []
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        for ids, seq_mask in zip(input_ids_list, seq_mask_list):
            pad_len = max_len - len(ids)
            padded_ids = F.pad(ids, (0, pad_len), value=pad_token_id)
            padded_seq_mask = F.pad(seq_mask, (0, pad_len), value=False) 
            
            padded_input_ids.append(padded_ids)
            padded_seq_masks.append(padded_seq_mask)
            
        return {
            "janus_input_ids": torch.stack(padded_input_ids),
            "janus_pixel_values": torch.stack([x['janus_pixel_values'] for x in batch]),
            "janus_images_seq_mask": torch.stack(padded_seq_masks),
            "janus_images_emb_mask": torch.stack([x['janus_images_emb_mask'] for x in batch]),
            "actions": torch.stack([x['actions'] for x in batch]),                     
            "videos": torch.stack([x['videos'] for x in batch])                        
        }

def save_checkpoint(
    model,
    processor,
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    stats_data: dict = None
) -> None:
    
    save_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}-step-{global_step}")
    
    if accelerator.is_main_process:
        if hasattr(args, 'max_ckpts') and args.max_ckpts > 0:
            checkpoint_dirs = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
            if len(checkpoint_dirs) >= args.max_ckpts:
                oldest_ckpt = min(checkpoint_dirs, key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                shutil.rmtree(os.path.join(args.output_dir, oldest_ckpt))
                logger.info(f"Removed old checkpoint: {oldest_ckpt}")

        os.makedirs(save_dir, exist_ok=True)

        full_state_dict = accelerator.get_state_dict(model)
        torch.save(full_state_dict, os.path.join(save_dir, "cosmos_janus_mot.pt"))

        processor.save_pretrained(save_dir)

        if stats_data is not None:
            stats_path = os.path.join(save_dir, 'train_statistics.json')
            with open(stats_path, 'w') as f:
                json.dump(stats_data, f, indent=2)
            logger.info(f"Statistics saved to {stats_path}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(f'Checkpoint {epoch}-{global_step} saved successfully to {save_dir}')

def train(args):
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        wandb.init(project=args.experiment_name, name=args.run_name, config=args, dir=args.log_dir)

    processor = VLChatProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    accelerator.print("Loading Janus Action Base...")
    janus_model = AutoModelForCausalLM.from_pretrained(
        args.action_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        flow=True, action_dim=args.action_dim, ignore_mismatched_sizes=True
    )
    
    accelerator.print("Loading Cosmos Video Base from Official Checkpoint...")
    experiment_opts = [
        "data_train=mock", 
        "data_val=mock",
    ]
    cosmos_wrapper, cosmos_config = load_model_from_checkpoint(
        experiment_name=args.cosmos_experiment_name,  
        s3_checkpoint_dir=args.cosmos_model_path,     
        config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        load_ema_to_reg=True,                         
        to_device=accelerator.device.type,
        experiment_opts=experiment_opts 
    )

    cosmos_dit = cosmos_wrapper.net
    cosmos_vae = cosmos_wrapper.tokenizer 
    
    accelerator.print("Building Cosmos-Janus MoT VLA...")
    model = CosmosJanusMoT(cosmos_dit, cosmos_vae, janus_model, args).to(accelerator.device, torch.bfloat16)

    trainable_params = []

    for param in model.parameters():
        param.requires_grad = True
    frozen_modules = [
        "janus.vision_model",
        "janus.aligner",
        "janus.gen_vision_model",
        "janus.gen_aligner",
        "cosmos_vae",
    ]
    
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in frozen_modules):
            param.requires_grad = False
        else:
            trainable_params.append(param)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    frozen_count = total_count - trainable_count
    
    accelerator.print("\n==== Parameter Freeze Status ====")
    accelerator.print(f"Total Params:     {total_count/1e9:.2f}B")
    accelerator.print(f"Trainable Params: {trainable_count/1e9:.2f}B ({(trainable_count/total_count)*100:.2f}%)")
    accelerator.print(f"Frozen Params:    {frozen_count/1e9:.2f}B")
    accelerator.print("Frozen Modules:   " + ", ".join(frozen_modules) + "\n")

    # no_decay = ["bias", "LayerNorm.weight", "norm.weight", "RMSNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    cosmos_params = []
    janus_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'cosmos_dit' in name:
            cosmos_params.append(param)
        else:
            janus_params.append(param)

    video_lr = args.learning_rate * 0.1 
    action_lr = args.learning_rate 

    optimizer_grouped_parameters = [
        {"params": cosmos_params, "lr": video_lr, "weight_decay": args.weight_decay},
        {"params": janus_params,  "lr": action_lr, "weight_decay": args.weight_decay},
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    train_dataset = VLADataset(args, processor, accelerator)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, collate_fn=train_dataset.collate_fn)

    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps
    lr_scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps, min_lr_ratio=args.min_lr_ratio
    )
    
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    model.train()
    global_step = 0

    video_frozen = False

    for epoch in range(args.n_epochs):
        # Freeze video backbone after specified epoch
        if not video_frozen and args.freeze_video_after >= 0 and epoch > args.freeze_video_after:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.freeze_video_backbone()
            video_frozen = True
            accelerator.print(f">>> Epoch {epoch}: Froze Cosmos video backbone. Only training action expert from now on.")

        train_iter = train_dataloader
        if accelerator.is_main_process:
            from tqdm import tqdm
            train_iter = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
            
        for batch in train_iter:
            with accelerator.accumulate(model):
                janus_input_ids = batch['janus_input_ids'].to(accelerator.device)
                janus_pixel_values = batch['janus_pixel_values'].to(accelerator.device).to(torch.bfloat16)
                janus_images_seq_mask = batch['janus_images_seq_mask'].to(accelerator.device)
                janus_images_emb_mask = batch['janus_images_emb_mask'].to(accelerator.device)
                actions = batch['actions'].to(accelerator.device)
                videos = batch['videos'].to(accelerator.device).to(torch.bfloat16)

                # print(videos.shape)
                # print(actions.shape)
                # print(janus_input_ids.shape)
                # torch.set_printoptions(profile="full")
                # print(janus_input_ids[0])
                # print(janus_pixel_values.shape) 
                # input("Check data shapes above. Press Enter to continue...")

                
                first_frame = videos[:, :, 0]
                video_frames = videos[:, :, 1:]
                
                video_loss_weight = 0.0 if video_frozen else 0.2
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss, v_loss, a_loss = model(
                        first_frame=first_frame,
                        video_frames=video_frames,
                        actions=actions,
                        janus_input_ids=janus_input_ids,
                        janus_pixel_values=janus_pixel_values,
                        janus_images_seq_mask=janus_images_seq_mask,
                        janus_images_emb_mask=janus_images_emb_mask,
                        fps=args.fps,
                        loss_weights=(video_loss_weight, 1.0) 
                    )
                
                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                lr_scheduler.step()
                global_step += 1
                if accelerator.is_main_process:
                    wandb.log({
                        'total_loss': loss.item(),
                        'video_loss': v_loss,
                        'action_loss': a_loss,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }, step=global_step)
                    train_iter.set_postfix(v_loss=f"{v_loss:.4f}", a_loss=f"{a_loss:.4f}")

        if ((epoch + 1) % args.save_freq == 0) or (epoch == args.n_epochs-1):
            accelerator.wait_for_everyone()
            save_checkpoint(
                model=model,
                processor=processor, 
                accelerator=accelerator,
                args=args,
                epoch=epoch,
                global_step=global_step,
                stats_data=train_dataset.stats_data,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # base config
    parser.add_argument('--experiment_name', type=str, default='cosmos_janus_mot')
    parser.add_argument('--run_name', type=str, default='run_1')
    parser.add_argument('--model_path', type=str, required=True, help='Janus Base Path')
    parser.add_argument('--action_model_path', type=str, required=True, help='Janus Action Expert Path')
    
    # data config
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--log_dir', type=str, default='./logs')
    
    # videos
    parser.add_argument('--video_h', type=int, default=256)
    parser.add_argument('--video_w', type=int, default=256)
    parser.add_argument('--video_frames', type=int, default=16)
    parser.add_argument('--fps', type=int, default=10)
    
    # Cosmos 
    parser.add_argument('--cosmos_model_path', type=str, required=True, help='Path to Cosmos .pt checkpoint')
    parser.add_argument('--cosmos_experiment_name', type=str, default='Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only', help='The registered experiment config name for Cosmos 2B/14B')
    
    # training hyperparameters
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--train_bsz_per_gpu', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_lr_ratio', type=float, default=0.05)
    parser.add_argument('--warmup_rates', type=float, default=0.05)
    parser.add_argument('--robot_state', type=int, default=0)
    parser.add_argument('--action_dim', type=int, default=14)
    parser.add_argument('--action_chunk', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--action_intermediate_size', type=int, default=0,
                        help='If >0, use a slim MLP with this intermediate size for the action expert bridges')
    parser.add_argument('--share_video_action_timestep', type=int, default=0,
                        help='If 1, reuse the same sampled timestep for the video and action branches')
    parser.add_argument('--freeze_video_after', type=int, default=-1,
                        help='Freeze Cosmos video backbone after this epoch (0-indexed). -1 = never freeze.')

    args = parser.parse_args()
    
    args.log_dir = os.path.join(args.log_dir, args.run_name)
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    
    train(args)
