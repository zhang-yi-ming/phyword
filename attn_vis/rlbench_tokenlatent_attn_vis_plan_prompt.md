# Beta RLBench Token-Latent Attention Visualization Prompt

你的任务是为这个训练脚本产出的 beta RLBench keyframe ckpt 做 attention 可视化：

`/mnt/nas/zhangyiming/last05_beta/last05/scripts/train_cot_2expert_rlbench_keyframe.sh`

请先进入只读调研，理解下面这些文件，再给出实现计划；实现时只能在下面目录内新增或修改文件：

`/mnt/nas/zhangyiming/last05_beta/last05/attn_vis`

严禁修改该目录外的任何文件。可以读取、复制、参考目录外文件，但复制后的改动只能落在 `attn_vis` 目录内。

## 参考文件

1. beta RLBench keyframe eval shell：
   `/mnt/nas/zhangyiming/last05_beta/last05/experiments/test_rlbench_keyframe_2expert.sh`

2. beta RLBench keyframe eval python：
   `/mnt/nas/zhangyiming/last05_beta/last05/experiments/robot/rlbench/run_rlbench_eval_keyframe.py`

3. beta LIBERO attention visualization shell：
   `/mnt/nas/zhangyiming/last05_beta/last05/experiments/test_libero_attn_vis.sh`

4. beta LIBERO attention visualization python：
   `/mnt/nas/zhangyiming/last05_beta/last05/experiments/robot/libero/run_libero_eval_attn_vis.py`

5. beta RLBench train python，重点看训练集 JSON 如何加载和构造模型输入：
   `/mnt/nas/zhangyiming/last05_beta/last05/scripts/train_cot_rlbench_keyframe.py`

## 目标

在 `/mnt/nas/zhangyiming/last05_beta/last05/attn_vis` 下创建一个可运行的 shell 脚本，必要时再创建一个 Python 文件。这个工具不是 RLBench 环境评测，不要和环境交互，不要 `env.reset` / `env.step`。它要在 eval / inference 模式下全量加载 `train_cot_2expert_rlbench_keyframe.sh` 训练出来的 ckpt，然后直接读取训练 JSON 中每个 task 的前 N 个 trajectory / episode 的全部 record，逐条做模型推理，并保存 attention 可视化图。

这版 beta 模型和 last05_beta 的连续 future image/state latent 不一样。它是 token-latent 版本：训练时只有 1 个 latent token，没有 future image latent，也没有 future state latent。实现时必须以 beta 训练脚本为准。

## 必须对齐的训练配置

训练脚本中的默认配置必须对齐：

- `LAST05_ROOT=/mnt/nas/zhangyiming/last05_beta/last05`
- `OUTPUT_ROOT_DIR=/mnt/nas/zhangyiming/last05_beta/last05/exp_cosmos_vla_3expert_rlbench_keyframe`
- `RUN_NAME=cosmos2B_janus1B_2expert_rlbench_keyframe_tokenlatent_hidden_300ep_1chunk`
- `DATA_JSON=/mnt/nas/zhangyiming/database/rlbench/train/json/train_action_chunk1_sumpos_lastrot.json`
- `JANUS_MODEL_PATH=/mnt/nas/zhangyiming/database/ckpt/pretrained/Janus-Pro-1B`
- `ACTION_MODEL_PATH=/mnt/nas/zhangyiming/database/ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr`
- `COSMOS_MODEL_PATH=/mnt/nas/zhangyiming/database/ckpt/pretrained/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt`
- `COSMOS_EXPERIMENT_NAME=Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only`
- `ACTION_DIM=7`
- `ACTION_CHUNK=1`
- `VIDEO_FRAMES=5`
- `NUM_COND_INPUT_FRAMES=1`
- `NUM_FUTURE_FRAMES=0`
- `IMG_LATENTS_PER_FUTURE=0`
- `STATE_LATENTS_PER_FUTURE=0`
- `TOTAL_LATENT_TOKENS=1`
- `FUTURE_FRAME_STRIDE=1`
- `ROBOT_STATE=0`
- `STATE_PLACEHOLDER_TOKENS=1`
- `STATE_DIM=7`
- `STATE_ENCODING_MODE=mlp`
- `ACTION_INTERMEDIATE_SIZE=5632`
- `VIDEO_LOSS_WEIGHT=0`
- `LATENT_LOSS_WEIGHT=1.0`
- `USE_LATENT_HIDDEN_SIM_LOSS=1`
- `LATENT_HIDDEN_SIM_LOSS_WEIGHT=1.0`
- `DECOSMOS=true`
- `TRAIN_EMBED_TOKENS=1`
- `ACTION_USE_LATENT_PREFIX=true`
- `COSMOS_SELF_ONLY_BRIDGE=true`
- `BRIDGE_POS_SCHEME=llama1d`
- `ACTION_SELF_CAUSAL_IN_BRIDGE=true`
- `video_h=32`
- `video_w=32`
- `fps=20`
- `ACTION_DENOISE_STEPS` 默认 10
- `COSMOS_DENOISE_STEPS` 默认 2

注意：`/mnt/nas/zhangyiming/last05_beta/last05/experiments/test_rlbench_keyframe_2expert.sh` 里的默认 `RUN_NAME` 可能是旧的 `cosmos2B_janus1B_2expert_rlbench_keyframe_tokenlatent_100ep_1chunk`。本任务要可视化训练脚本产出的 ckpt，因此默认 `RUN_NAME` 必须以训练脚本的 `cosmos2B_janus1B_2expert_rlbench_keyframe_tokenlatent_hidden_300ep_1chunk` 为准，除非用户显式通过环境变量覆盖。

## 输入数据要求

- 读取训练 JSON：`/mnt/nas/zhangyiming/database/rlbench/train/json/train_action_chunk1_sumpos_lastrot.json`
- 按 task 分组。task 可以优先用 `record["task"]`，没有的话参考 `train_cot_rlbench_keyframe.py` 里的 `resolve_rlbench_episode_key` / `task_name_from_train_record`，从 `front_pic` 路径推断。
- 每个 task 取前 N 个 trajectory / episode，N 通过 shell 环境变量传入，例如 `NUM_TRAJECTORIES_PER_TASK`，默认可以设为 1。
- 对选中的每个 episode，按 `record_index` 排序，处理该 episode 的全部 records。
- 每个 record 的输入构造要尽量复用训练时 `VLACotDataset` 的逻辑：
  - prompt 使用 `sample["input_prompt"]`
  - 图片使用 `sample["front_pic"]`
  - Cosmos 条件视频用训练里的 `_load_video` / `clipped_keyframe_indices` 逻辑，`VIDEO_FRAMES=5`，`NUM_COND_INPUT_FRAMES=1`，Resize+CenterCrop 到 `32x32`
  - Janus main image 使用 `front_pic` 原图，`processor(prompt=..., images=[front_image])`
  - `janus_action_pixel_values` 也用 `front_pic`，同训练逻辑
  - `robot_state=0`，因此不需要 current state token
  - beta token-latent 训练没有 future image/state latent 输入参与可视化推理，不要构造连续 future latent 监督
  - 推理调用 `model.forward_flow_joint_inference`，和 beta RLBench eval 的 predict path 保持一致

## Attention 可视化要求

- 参考 beta `run_libero_eval_attn_vis.py` 的 `AttentionMapRecorder` 思路，在 runtime monkey patch，不要修改 `models/cosmos_janus_cot.py`。
- 要抓 transformer / MoT 的每一层，通常是 `model.mot_attention_wrappers` 的全部层。
- 对每层中 token-latent 分支的 spatial latent token，以及 action token，作为 query，可视化它们对 latent branch 图片 token 的 attention。
- 这里 beta RLBench ckpt 是 `TOTAL_LATENT_TOKENS=1`、`ACTION_CHUNK=1`，所以不要照搬 LIBERO 可视化里 action chunk 16 的限制；要泛化成 1 个 spatial/latent token + 1 个 action token。
- 图片 token 是 Janus main-image token，对应 `janus_images_seq_mask`，期望数量是 576，即 `24x24`。如果不是 576，要报清楚错误。
- 注意力计算方式要和 beta LIBERO attn vis 保持一致：
  - 在 patched `forward_action_only` 或相同 runtime hook 中拿 latent/action 的 `q/k`
  - 选择 spatial latent query row 和 action query row
  - 选择 Janus image token 对应的 key columns
  - `scores = q @ k.T / sqrt(head_dim)`
  - 只在这 576 个 image token 内部做 `softmax(dim=-1)`
  - 多 head 取 mean
  - reshape 成 `24x24`
  - 上采样叠加到原图
- overlay 逻辑、颜色映射、透明度要照搬 beta LIBERO：
  - alpha 默认 `0.45`
  - colormap 使用 beta LIBERO attn vis 里的 red/green/blue 公式
  - heatmap 叠加前先做 min-max 到 0..1
  - 每张总图按行展示 layer，按列展示 `spatial` 或 `latent_00`，以及 `action_00`
- 建议支持 `ATTN_VIS_CAPTURE_MODE=all/first/last`，默认 `all`。如果 beta LIBERO 原实现没有这个参数，也可以在 RLBench trainset 可视化里补上，但只在 `attn_vis` 目录内实现。
- 输出目录建议：
  `/mnt/nas/zhangyiming/last05_beta/experiments_rlbench/attn_visualizations/${timestamp}_${run_name}/task=.../episode=.../`
  也可以由 `ATTN_VIS_DIR` 或 `ATTENTION_VISUALIZATION_DIR` 环境变量覆盖。
- 文件名要包含 task、episode_index、record_index、denoise step，便于定位。

## 实现要求

- 在 `attn_vis` 下至少创建：
  - 一个 shell，比如 `test_rlbench_trainset_attn_vis.sh`
  - 一个 Python，比如 `run_rlbench_trainset_attn_vis.py`
- shell 负责设置环境、checkpoint 路径、默认超参、输出目录、日志和 hparams 文件。
- Python 负责加载 ckpt、读取训练 JSON、选择每个 task 前 N 个 episode、逐 record 推理并保存可视化。
- ckpt 选择逻辑参考 beta RLBench eval：支持 `PRETRAINED_CHECKPOINT`；否则用 `OUTPUT_ROOT_DIR/RUN_NAME/CHECKPOINT_NAME`；`CHECKPOINT_NAME` 为空时自动找最新 checkpoint。
- 必须 full load checkpoint，和 beta `run_rlbench_eval_keyframe.py` 的 `model_load` 方式一致。
- 不要启动 RLBench 环境，不要依赖 CoppeliaSim，不需要 Xvfb。
- 不要改任何 `attn_vis` 目录外的文件。

## 验收

- 先给出清晰计划，说明会新增哪些文件、每个文件负责什么。
- 实现后至少做：
  - `bash -n` 新 shell
  - `python -m py_compile` 新 Python
- 如果环境允许，可以用 `NUM_TRAJECTORIES_PER_TASK=1`、最多少量 records 做 smoke test；如果不实际跑大模型，也要说明原因。
- 最终汇报新增文件、运行命令、输出目录、以及哪些关键超参与训练脚本保持一致。
