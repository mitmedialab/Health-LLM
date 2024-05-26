#!/bin/bash
            
data_path="data/finetune_data.json"
output_path="medalpaca-13b"

torchrun --nproc_per_node=4 --master_port=2023 medalpaca/train.py \
    --model "llama-13b-bf" \
    --data_path "$data_path" \
    --output_dir "$output_path" \
    --train_in_8bit False \
    --use_lora False \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --gradient_checkpointing True \
    --global_batch_size 128 \
    --per_device_batch_size 4 \
    --num_epochs 5
