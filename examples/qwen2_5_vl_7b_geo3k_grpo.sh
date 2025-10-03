#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/root/code/MedicalVideoChat/Qwen2.5-VL/qwen-vl-finetune/fine_tuned_models/qwen2.5vl-7b-medvideo_09_03_visual_timestamp_prompt_v2/checkpoint-1800  # replace it with your local file path
DATA_DIR=/root/code/geometry3k/data  # replace if your geometry3k clone lives elsewhere

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_DIR}@train \
    data.val_files=${DATA_DIR}@validation \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
    trainer.n_gpus_per_node=2
