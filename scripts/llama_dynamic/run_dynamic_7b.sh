#!/bin/bash
cd /common-data/zhanghaojie/DropLora
# 5e-6 6e-6 7e-6 8e-6 9e-6 1e-5 2e-5 3e-5 4e-5 5e-5 6e-5 7e-5 8e-5 9e-5 1e-4

## cms
## droplora r=32,a=64
for lr in 3e-4; do
    echo "Running with lr=${lr}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    swift sft \
        --template llama \
        --use_chat_template false \
        --model /common-data/pretrained_models/llama2/llama-2-7b-hf/ \
        --dataset /common-data/zhanghaojie/DropLora/datasets/commonsense/train.json \
        --split_dataset_ratio 0 \
        --per_device_train_batch_size 16 \
        --num_train_epochs 3 \
        --torch_dtype bfloat16 \
        --train_type droplora \
        --lora_rank 32 \
        --lora_alpha 64 \
        --lora_dropout 0.05 \
        --droplora_dynamic_pruning true \
        --target_modules q_proj,k_proj,v_proj,up_proj,down_proj  \
        --output_dir output-cms3-7b-dynamic/${lr} \
        --add_version true \
        --learning_rate ${lr} \
        --gradient_checkpointing true \
        --val_dataset /common-data/zhanghaojie/DropLora/datasets/commonsense/test.json \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --lr_scheduler_type linear \
        --warmup_steps 100  \
        --attn_impl flash_attn \
        --gradient_accumulation_steps 2 \
        --save_only_model true \
        --deepspeed /common-data/zhanghaojie/DropLora/deepspeed/ds_z2_config.json
done

## math
## droplora r=32,a=64
for lr in 3e-4; do
    echo "Running with lr=${lr}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    swift sft \
        --template llama \
        --use_chat_template false \
        --model /common-data/pretrained_models/llama2/llama-2-7b-hf/ \
        --dataset /common-data/zhanghaojie/DropLora/datasets/metamath/train.json \
        --split_dataset_ratio 0 \
        --per_device_train_batch_size 16 \
        --num_train_epochs 1 \
        --torch_dtype bfloat16 \
        --train_type droplora \
        --lora_rank 64 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --droplora_dynamic_pruning true \
        --target_modules q_proj,k_proj,v_proj,up_proj,down_proj  \
        --output_dir output-metamath-7b-dynamic/${lr} \
        --add_version true \
        --learning_rate ${lr} \
        --gradient_checkpointing true \
        --val_dataset /common-data/zhanghaojie/DropLora/datasets/metamath/test.json \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --lr_scheduler_type linear \
        --warmup_steps 100  \
        --attn_impl flash_attn \
        --gradient_accumulation_steps 2 \
        --save_only_model true \
        --deepspeed /common-data/zhanghaojie/DropLora/deepspeed/ds_z2_config.json
done

# python
# droplora r=32,a=64
for lr in 3e-4; do
    echo "Running with lr=${lr}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    swift sft \
        --template llama \
        --use_chat_template false \
        --model /common-data/pretrained_models/llama2/llama-2-7b-hf/ \
        --dataset /common-data/zhanghaojie/DropLora/datasets/python/train.json \
        --split_dataset_ratio 0 \
        --per_device_train_batch_size 16 \
        --num_train_epochs 3 \
        --torch_dtype bfloat16 \
        --train_type droplora \
        --lora_rank 32 \
        --lora_alpha 64 \
        --lora_dropout 0.05 \
        --droplora_dynamic_pruning true \
        --target_modules q_proj,k_proj,v_proj,up_proj,down_proj  \
        --output_dir output-python-7b-dynamic/${lr} \
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
        --gradient_accumulation_steps 2 \
        --save_only_model true \
        --deepspeed /common-data/zhanghaojie/DropLora/deepspeed/ds_z2_config.json
done

## conversation
## droplora r=32,a=64
for lr in 3e-4; do
    echo "Running with lr=${lr}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    swift sft \
        --template llama \
        --use_chat_template false \
        --model /common-data/pretrained_models/llama2/llama-2-7b-hf/ \
        --dataset /common-data/zhanghaojie/DropLora/datasets/conversations/train.json \
        --split_dataset_ratio 0 \
        --per_device_train_batch_size 16 \
        --num_train_epochs 3 \
        --torch_dtype bfloat16 \
        --train_type droplora \
        --lora_rank 32 \
        --lora_alpha 64 \
        --lora_dropout 0.05 \
        --droplora_dynamic_pruning true \
        --target_modules q_proj,k_proj,v_proj,up_proj,down_proj  \
        --output_dir output-conversations-7b-dynamic/${lr} \
        --add_version true \
        --learning_rate ${lr} \
        --gradient_checkpointing true \
        --val_dataset /common-data/zhanghaojie/DropLora/datasets/conversations/test.json \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --lr_scheduler_type linear \
        --warmup_steps 100  \
        --attn_impl flash_attn \
        --gradient_accumulation_steps 2 \
        --save_only_model true \
        --deepspeed /common-data/zhanghaojie/DropLora/deepspeed/ds_z2_config.json
done