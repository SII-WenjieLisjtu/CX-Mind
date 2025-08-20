#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER=GNU

ray stop
GPU_NUM=8

ray start --head --node-ip-address 127.0.0.1 --num-gpus ${GPU_NUM} --port 8262

MODEL_PATH=/mnt/lustrenew/sunhaoran/Model/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$GPU_NUM
