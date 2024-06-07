#!/bin/bash


echo "########### TRAINING (TASK-1) ###########"

# Evaluation: vinai/phobert-base 
python run_evaluation_cls_task.py --task "task-1" --model_type "simple" --model_name "vinai/phobert-base" --source_len 200 --batch_size 16
python run_evaluation_cls_task.py --task "task-1" --model_type "simple" --model_name "uitnlp/visobert" --source_len 400 --batch_size 16



echo "All commands completed!"