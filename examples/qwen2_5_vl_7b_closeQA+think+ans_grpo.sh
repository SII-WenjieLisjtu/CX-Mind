#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER=GNU

ray stop
GPU_NUM=4

ray start --head --node-ip-address 127.0.0.1 --num-gpus ${GPU_NUM} --port 8262

# MODEL_PATH=xxx  # replace it with your local file path
# MODEL_PATH=xxx
MODEL_PATH=xxx

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=./CX-Mind/data/closeqa_RL_train.jsonl@train \
    data.val_files=./CX-Mind/data/closeqa_RL_val.jsonl \
    data.max_response_length=4096 \
    data.rollout_batch_size=64 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=10 \
    worker.reward.reward_function=./examples/reward_function/closeQA+think+ans.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_7b_closeQA+think+ans_grpo_1epoch \
    trainer.total_epochs=1 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$GPU_NUM