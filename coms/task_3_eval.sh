#!/bin/bash

# Redirect outputs
exec > >(tee -i ./results/logs/task_3_eval.log) 2>&1



# vinai/bartpho-syllable-based

echo "Evaluating vinai/bartpho-syllable-base"
python run_evaluation_generation_task.py --task "task-3" --model_name "vinai/bartpho-syllable-base" --source_len 768 --target_len 128 --batch_size 12


# Training vinai/bartpho-word-base

echo "Evaluating vinai/bartpho-word-base"
python run_evaluation_generation_task.py --task "task-3" --model_name "vinai/bartpho-word-base" --source_len 768 --target_len 128 --batch_size 12


# VietAI/vit5-base

echo "Evaluating VietAI/vit5-base"
python run_evaluation_generation_task.py --task "task-3" --model_name "VietAI/vit5-base" --source_len 768 --target_len 128 --batch_size 12


echo "All commands completed!"