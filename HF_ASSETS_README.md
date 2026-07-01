# PhysicalWord last05_mot2_action Assets

This dataset repo contains the private/custom assets needed to train
`last05_mot2_action` on a machine that cannot access the source NAS.

Code repo expected by the migration guide:

- GitHub: `https://github.com/zhang-yi-ming/phyword.git`
- Branch: `mot2_cosmos_ae`
- Commit: `1495a9d`

## Included

This repo intentionally includes:

```text
ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr/
data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/
rlbench/
```

Approximate source sizes:

```text
ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr                         4.0G
data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual  2.8G
rlbench                                                               25G
```

## Not Included

Cosmos Predict2.5 weights are not included in this repo. Download them from
the official gated NVIDIA Hugging Face model after accepting the license:

```bash
huggingface-cli download nvidia/Cosmos-Predict2.5-2B \
  --local-dir /mnt/nas/zhangyiming/database/ckpt/pretrained/Cosmos-Predict2.5-2B
```

The source scripts expect the Cosmos checkpoint at:

```text
/mnt/nas/zhangyiming/database/ckpt/pretrained/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt
```

Already-trained MoT2 evaluation checkpoints are also not included.

## Target Layout

Download this dataset repo directly into:

```bash
/mnt/nas/zhangyiming/database
```

Example:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /mnt/nas/zhangyiming/database

huggingface-cli download <owner>/physicalword-assets \
  --repo-type dataset \
  --local-dir /mnt/nas/zhangyiming/database
```

The training JSON files contain absolute paths under
`/mnt/nas/zhangyiming/database`. If the assets are downloaded elsewhere, create
a symlink:

```bash
mkdir -p /mnt/nas/zhangyiming
ln -sfn /actual/database/path /mnt/nas/zhangyiming/database
```

## Asset Notes

### Action Expert

Expected target path:

```text
/mnt/nas/zhangyiming/database/ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr
```

Files:

```text
config.json
model.safetensors
preprocessor_config.json
processor_config.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json
```

This directory is used as both the action expert and processor/tokenizer source
by the MoT2 scripts. Do not replace it with Janus-Pro-1B.

### RLBench

Expected target root:

```text
/mnt/nas/zhangyiming/database/rlbench
```

The full source `rlbench` directory is included. The key training files used by
the current scripts include:

```text
rlbench/train/json/train_action_chunk1_sumpos_lastrot.json
rlbench/train/json/train_action_chunk1_sumpos_lastrot_statistics.json
rlbench/train/json/cosmos_text_cache_rlbench_keyframe/
rlbench/train/images/
```

The JSON references image paths such as:

```text
/mnt/nas/zhangyiming/database/rlbench/train/images/...
```

### LIBERO

Expected target path:

```text
/mnt/nas/zhangyiming/database/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual
```

Included files/directories:

```text
train_with_atomic_action.json
train.json
train_rewritten_prompts.json
train_statistics.json
videos/
cosmos_text_cache/
cosmos_text_cache_raw_full_concat/
cosmos_text_cache_rewritten_prompts_raw_full_concat/
```

The JSON references video paths such as:

```text
/mnt/nas/zhangyiming/database/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/videos/...
```

## Main Training Entrypoints

RLBench:

```bash
cd /mnt/nas/zhangyiming/last05_beta/last05_mot2_action
bash scripts/train_mot2_rlbench_keyframe.sh
```

LIBERO:

```bash
cd /mnt/nas/zhangyiming/last05_beta/last05_mot2_action
bash scripts/train_mot2.sh
```
