#!/bin/bash

# Redirect outputs
exec > output.log 2>&1

echo "### Training "
# Evaluation: vinai/phobert-base 
echo "Task 1: vinai/phobert-base"
python run_evaluation_cls_task.py --task "task-1" --model_type "simple" --model_name "vinai/phobert-base" --source_len 200 --batch_size 16
echo "Task 2: vinai/phobert-base"
python run_evaluation_cls_task.py --task "task-2" --model_type "simple" --model_name "vinai/phobert-base" --source_len 200 --batch_size 16

echo "All commands completed!"