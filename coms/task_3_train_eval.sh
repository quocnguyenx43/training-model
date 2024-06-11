#!/bin/bash

# Redirect outputs
exec > >(tee -i ./results/logs/task_3_train_eval.log) 2>&1


# Training
echo "Training VietAI/vit5-base"
python run_train_generation_task.py --task "task-3" --model_name "VietAI/vit5-base" --source_len 768 --target_len 128 --batch_size 12 --learning_rate 0.001 --epochs 3

echo "Evaluating VietAI/vit5-base"
python run_evaluation_generation_task.py --task "task-3" --model_name "VietAI/vit5-base" --source_len 768 --target_len 128 --batch_size 12

echo "All commands completed!"
