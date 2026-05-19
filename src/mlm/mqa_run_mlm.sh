#!/bin/bash

uv add adapters evaluate
uv add accelerate -U

languages=('tha-Thai')
adapter_sources=("glot")

export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=""

BASE_DIR=~/hihi/CS221/Knowledge-Driven-Adaptation-LLMs

for source in "${adapter_sources[@]}"; do
    echo "Using adapter source: $source"

    for lang in "${languages[@]}"; do
        echo "Training for language: $lang"

        torchrun --nproc_per_node=1 run_mlm.py \
            --model_name_or_path google-bert/bert-base-multilingual-cased \
            --train_file "${BASE_DIR}/data/$source/train_glot_${lang}.csv" \
            --validation_file "${BASE_DIR}/data/$source/val_glot_${lang}.csv" \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --do_train \
            --do_eval \
            --logging_dir "${BASE_DIR}/outputs/logs/${lang}" \
            --output_dir "${BASE_DIR}/outputs/models/${lang}" \
            --train_adapter \
            --learning_rate 1e-4 \
            --adapter_config seq_bn_inv \
            --overwrite_output_dir \
            --save_total_limit=1 \
            --evaluation_strategy steps \
            --save_strategy steps \
            --max_steps 1000 \
            --line_by_line

        echo "Training for $lang completed."
    done
done