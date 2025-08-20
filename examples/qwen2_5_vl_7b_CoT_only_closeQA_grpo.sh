#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER=GNU

ray stop
GPU_NUM=8

ray start --head --node-ip-address 127.0.0.1 --num-gpus ${GPU_NUM} --port 8262

# MODEL_PATH=/mnt/lustrenew/sunhaoran/Model/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
# MODEL_PATH=/mnt/lustrenew/sunhaoran/train/x-ray/stage3/qwen2_5vl-7b/checkpoint-32332-merged
MODEL_PATH=/mnt/lustrenew/sunhaoran/train/x-ray/stage3/qwen2_5vl-7b/v2-20250716-192432/checkpoint-3975-merged

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/mnt/lustrenew/liwenjie/EasyR1-data/closeqa_RL_train.jsonl@train \
    data.val_files=/mnt/lustrenew/liwenjie/EasyR1-data/closeqa_RL_val.jsonl \
    data.max_response_length=4096 \
    data.rollout_batch_size=64 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=10 \
    worker.reward.reward_function=./examples/reward_function/closeQA.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_7b_CoT_only_closeQA_grpo_1epoch \
    trainer.total_epochs=1 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.load_checkpoint_path=/mnt/lustrenew/liwenjie/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_CoT_only_closeQA_grpo_1epoch/global_step_120
    # data.val_files=/mnt/lustrenew/liwenjie/EasyR1-data/openqa_RL.jsonl@test \