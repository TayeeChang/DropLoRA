#!/bin/bash
cd /common-data/zhanghaojie/DropLora
# 5e-6 6e-6 7e-6 8e-6 9e-6 1e-5 2e-5 3e-5 4e-5 5e-5 6e-5 7e-5 8e-5 9e-5 1e-4

# 1
# droplora r=32,a=64
for lr in 3e-4; do
    echo "Running with lr=${lr}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    swift sft \
        --template llama \
        --use_chat_template false \
        --model /common-data/pretrained_models/llama3-8b/LLM-Research/Meta-Llama-3-8B/ \
        --dataset /common-data/zhanghaojie/DropLora/datasets/python/train.json \
        --split_dataset_ratio 0 \
        --per_device_train_batch_size 4 \
        --num_train_epochs 3 \
        --torch_dtype bfloat16 \
        --train_type droplora \
        --lora_rank 32 \
        --lora_alpha 64 \
        --lora_dropout 0.05 \
        --target_modules q_proj,k_proj,v_proj,up_proj,down_proj  \
        --output_dir output-python-8b/${lr} \
        --add_version true \
        --learning_rate ${lr} \
        --gradient_checkpointing true \
        --val_dataset /common-data/zhanghaojie/DropLora/datasets/python/test.json \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --lr_scheduler_type linear \
        --warmup_steps 100  \
        --attn_impl flash_attn \
        --gradient_accumulation_steps 8 \
        --save_only_model true \
        --deepspeed /common-data/zhanghaojie/DropLora/deepspeed/ds_z2_config.json
done

## 2
# dora r=32,a=64
for lr in 3e-4; do
    echo "Running with lr=${lr}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    swift sft \
        --template llama \
        --use_chat_template false \
        --model /common-data/pretrained_models/llama3-8b/LLM-Research/Meta-Llama-3-8B/ \
        --dataset /common-data/zhanghaojie/DropLora/datasets/python/train.json \
        --split_dataset_ratio 0 \
        --per_device_train_batch_size 4 \
        --num_train_epochs 3 \
        --torch_dtype bfloat16 \
        --train_type lora \
        --use_dora true \
        --lora_rank 32 \
        --lora_alpha 64 \
        --lora_dropout 0.05 \
        --target_modules q_proj,k_proj,v_proj,up_proj,down_proj  \
        --output_dir output-python-8b/${lr} \
        --add_version true \
        --learning_rate ${lr} \
        --gradient_checkpointing true \
        --val_dataset /common-data/zhanghaojie/DropLora/datasets/python/test.json \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --lr_scheduler_type linear \
        --warmup_steps 100  \
        --attn_impl flash_attn \
        --gradient_accumulation_steps 8 \
        --save_only_model true \
        --deepspeed /common-data/zhanghaojie/DropLora/deepspeed/ds_z2_config.json
done

## 3
# lora r=32,a=64
for lr in 3e-4; do
    echo "Running with lr=${lr}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    swift sft \
        --template llama \
        --use_chat_template false \
        --model /common-data/pretrained_models/llama3-8b/LLM-Research/Meta-Llama-3-8B/ \
        --dataset /common-data/zhanghaojie/DropLora/datasets/python/train.json \
        --split_dataset_ratio 0 \
        --per_device_train_batch_size 4 \
        --num_train_epochs 3 \
        --torch_dtype bfloat16 \
        --train_type lora \
        --lora_rank 32 \
        --lora_alpha 64 \
        --lora_dropout 0.05 \
        --target_modules q_proj,k_proj,v_proj,up_proj,down_proj  \
        --output_dir output-python-8b/${lr} \
        --add_version true \
        --learning_rate ${lr} \
        --gradient_checkpointing true \
        --val_dataset /common-data/zhanghaojie/DropLora/datasets/python/test.json \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --lr_scheduler_type linear \
        --warmup_steps 100  \
        --attn_impl flash_attn \
        --gradient_accumulation_steps 8 \
        --save_only_model true \
        --deepspeed /common-data/zhanghaojie/DropLora/deepspeed/ds_z2_config.json
done

## 4
## pissa r=32,a=32
for lr in 3e-4; do
    echo "Running with lr=${lr}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    swift sft \
        --template llama \
        --use_chat_template false \
        --model /common-data/pretrained_models/llama3-8b/LLM-Research/Meta-Llama-3-8B/ \
        --dataset /common-data/zhanghaojie/DropLora/datasets/python/train.json \
        --split_dataset_ratio 0 \
        --per_device_train_batch_size 4 \
        --num_train_epochs 3 \
        --torch_dtype bfloat16 \
        --train_type lora \
        --init_weights pissa \
        --lora_rank 32 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --target_modules q_proj,k_proj,v_proj,up_proj,down_proj  \
        --output_dir output-python-8b/${lr} \
        --add_version true \
        --learning_rate ${lr} \
        --gradient_checkpointing true \
        --val_dataset /common-data/zhanghaojie/DropLora/datasets/python/test.json \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --lr_scheduler_type linear \
        --warmup_steps 100  \
        --attn_impl flash_attn \
        --gradient_accumulation_steps 8 \
        --save_only_model true \
        --deepspeed /common-data/zhanghaojie/DropLora/deepspeed/ds_z2_config.json
done

## 5
## milora r=32,a=32
for lr in 3e-4; do
    echo "Running with lr=${lr}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    swift sft \
        --template llama \
        --use_chat_template false \
        --model /common-data/pretrained_models/llama3-8b/LLM-Research/Meta-Llama-3-8B/ \
        --dataset /common-data/zhanghaojie/DropLora/datasets/python/train.json \
        --split_dataset_ratio 0 \
        --per_device_train_batch_size 4 \
        --num_train_epochs 3 \
        --torch_dtype bfloat16 \
        --train_type lora \
        --init_weights milora \
        --lora_rank 32 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --target_modules q_proj,k_proj,v_proj,up_proj,down_proj  \
        --output_dir output-python-8b/${lr} \
        --add_version true \
        --learning_rate ${lr} \
        --gradient_checkpointing true \
        --val_dataset /common-data/zhanghaojie/DropLora/datasets/python/test.json \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --lr_scheduler_type linear \
        --warmup_steps 100  \
        --attn_impl flash_attn \
        --gradient_accumulation_steps 8 \
        --save_only_model true \
        --deepspeed /common-data/zhanghaojie/DropLora/deepspeed/ds_z2_config.json
done

# 6
for lr in 3e-4; do
    echo "Running with lr=${lr}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    swift sft \
        --template llama \
        --use_chat_template false \
        --model /common-data/pretrained_models/llama3-8b/LLM-Research/Meta-Llama-3-8B/ \
        --dataset /common-data/zhanghaojie/DropLora/datasets/python/train.json \
        --split_dataset_ratio 0 \
        --per_device_train_batch_size 4 \
        --num_train_epochs 3 \
        --torch_dtype bfloat16 \
        --train_type droplora \
        --lora_rank 32 \
        --lora_alpha 64 \
        --lora_dropout 0.05 \
        --droplora_inner_dropout 0.4 \
        --target_modules q_proj,k_proj,v_proj,up_proj,down_proj  \
        --output_dir output-python-8b/${lr} \
        --add_version true \
        --learning_rate ${lr} \
        --gradient_checkpointing true \
        --val_dataset /common-data/zhanghaojie/DropLora/datasets/python/test.json \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --lr_scheduler_type linear \
        --warmup_steps 100  \
        --attn_impl flash_attn \
        --gradient_accumulation_steps 8 \
        --save_only_model true \
        --deepspeed /common-data/zhanghaojie/DropLora/deepspeed/ds_z2_config.json
done

# 7
for lr in 3e-4; do
    echo "Running with lr=${lr}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    swift sft \
        --template llama \
        --use_chat_template false \
        --model /common-data/pretrained_models/llama3-8b/LLM-Research/Meta-Llama-3-8B/ \
        --dataset /common-data/zhanghaojie/DropLora/datasets/python/train.json \
        --split_dataset_ratio 0 \
        --per_device_train_batch_size 4 \
        --num_train_epochs 3 \
        --torch_dtype bfloat16 \
        --train_type droplora \
        --lora_rank 32 \
        --lora_alpha 64 \
        --lora_dropout 0.05 \
        --droplora_inner_dropout 0.3 \
        --target_modules q_proj,k_proj,v_proj,up_proj,down_proj  \
        --output_dir output-python-8b/${lr} \
        --add_version true \
        --learning_rate ${lr} \
        --gradient_checkpointing true \
        --val_dataset /common-data/zhanghaojie/DropLora/datasets/python/test.json \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --lr_scheduler_type linear \
        --warmup_steps 100  \
        --attn_impl flash_attn \
        --gradient_accumulation_steps 8 \
        --save_only_model true \
        --deepspeed /common-data/zhanghaojie/DropLora/deepspeed/ds_z2_config.json
done


# 8
for lr in 3e-4; do
    echo "Running with lr=${lr}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    swift sft \
        --template llama \
        --use_chat_template false \
        --model /common-data/pretrained_models/llama3-8b/LLM-Research/Meta-Llama-3-8B/ \
        --dataset /common-data/zhanghaojie/DropLora/datasets/python/train.json \
        --split_dataset_ratio 0 \
        --per_device_train_batch_size 4 \
        --num_train_epochs 3 \
        --torch_dtype bfloat16 \
        --train_type droplora \
        --lora_rank 32 \
        --lora_alpha 64 \
        --lora_dropout 0.05 \
        --droplora_inner_dropout 0.2 \
        --target_modules q_proj,k_proj,v_proj,up_proj,down_proj  \
        --output_dir output-python-8b/${lr} \
        --add_version true \
        --learning_rate ${lr} \
        --gradient_checkpointing true \
        --val_dataset /common-data/zhanghaojie/DropLora/datasets/python/test.json \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --lr_scheduler_type linear \
        --warmup_steps 100  \
        --attn_impl flash_attn \
        --gradient_accumulation_steps 8 \
        --save_only_model true \
        --deepspeed /common-data/zhanghaojie/DropLora/deepspeed/ds_z2_config.json
done


# 9
for lr in 3e-4; do
    echo "Running with lr=${lr}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    swift sft \
        --template llama \
        --use_chat_template false \
        --model /common-data/pretrained_models/llama3-8b/LLM-Research/Meta-Llama-3-8B/ \
        --dataset /common-data/zhanghaojie/DropLora/datasets/python/train.json \
        --split_dataset_ratio 0 \
        --per_device_train_batch_size 4 \
        --num_train_epochs 3 \
        --torch_dtype bfloat16 \
        --train_type droplora \
        --lora_rank 32 \
        --lora_alpha 64 \
        --lora_dropout 0.05 \
        --droplora_inner_dropout 0.1 \
        --target_modules q_proj,k_proj,v_proj,up_proj,down_proj  \
        --output_dir output-python-8b/${lr} \
        --add_version true \
        --learning_rate ${lr} \
        --gradient_checkpointing true \
        --val_dataset /common-data/zhanghaojie/DropLora/datasets/python/test.json \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --lr_scheduler_type linear \
        --warmup_steps 100  \
        --attn_impl flash_attn \
        --gradient_accumulation_steps 8 \
        --save_only_model true \
        --deepspeed /common-data/zhanghaojie/DropLora/deepspeed/ds_z2_config.json
done