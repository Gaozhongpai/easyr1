#!/bin/bash

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/root/code/Qwen2.5-VL/qwen-vl-finetune/fine_tuned_models/qwen2.5vl-7b-medvideo_09_03_visual_timestamp_prompt_v2/checkpoint-1800  # update with your checkpoint
DATA_DIR=/root/code/easyr1/medical_tal_rlhf_v2  # directory containing train.jsonl / validation.jsonl from prepare_medical_tal_dataset.py

export CUDA_VISIBLE_DEVICES=0,1
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_DIR}/train.jsonl \
    data.val_files=${DATA_DIR}/validation_small.jsonl \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.rollout_batch_size=32 \
    data.mini_rollout_batch_size=16 \
    data.val_batch_size=-1 \
    data.min_pixels=$((8 * 28 * 28)) \
    data.max_pixels=$((48 * 28 * 28)) \
    data.video_fps=2.0 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=32 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.ppo_epochs=1 \
    worker.rollout.n=4 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.max_num_batched_tokens=12288 \
    worker.reward.reward_type=batch \
    worker.reward.reward_function=./examples/reward_function/medical_tal.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_7b_medical_grpo \
    trainer.n_gpus_per_node=2 \
    trainer.val_generations_to_log=6 \
    trainer.save_freq=50 \
    trainer.val_before_train=false

