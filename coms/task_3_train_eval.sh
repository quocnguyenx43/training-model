#!/bin/bash

# Redirect outputs
exec > >(tee -i ./results/logs_raw/task_3_train_eval.log) 2>&1



# vinai/bartpho-syllable-based
echo "Training vinai/bartpho-syllable-base"
python run_train_generation_task.py --task "task-3" --model_name "vinai/bartpho-syllable-base" --source_len 768 --target_len 128 --batch_size 12 --learning_rate 0.001 --epochs 3

echo "Evaluating vinai/bartpho-syllable-base"
python run_evaluation_generation_task.py --task "task-3" --model_name "vinai/bartpho-syllable-base" --source_len 768 --target_len 128 --batch_size 12


# Training vinai/bartpho-word-base
echo "Training vinai/bartpho-word-base"
python run_train_generation_task.py --task "task-3" --model_name "vinai/bartpho-word-base" --source_len 768 --target_len 128 --batch_size 12 --learning_rate 0.001 --epochs 3

echo "Evaluating vinai/bartpho-word-base"
python run_evaluation_generation_task.py --task "task-3" --model_name "vinai/bartpho-word-base" --source_len 768 --target_len 128 --batch_size 12


# VietAI/vit5-base
echo "Training VietAI/vit5-base"
python run_train_generation_task.py --task "task-3" --model_name "VietAI/vit5-base" --source_len 768 --target_len 128 --batch_size 12 --learning_rate 0.001 --epochs 3

echo "Evaluating VietAI/vit5-base"
python run_evaluation_generation_task.py --task "task-3" --model_name "VietAI/vit5-base" --source_len 768 --target_len 128 --batch_size 12



echo "All commands completed!"

# vinai/bartpho-word-base
# vinai/bartpho-word
# vinai/bartpho-syllable-base
# vinai/bartpho-syllable

# VietAI/vit5-base
# VietAI/vit5-large