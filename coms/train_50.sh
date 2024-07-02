#!/bin/bash

# Redirect outputs
exec > >(tee -i ./results/logs_raw/train_50.log) 2>&1



echo " ========== TASK 1 TRAINING =========="
echo "lstm + distilbert-base-multilingual-cased"
python run_train_cls_task.py --task "task-1" --model_type "lstm" --hidden_size 128 --num_layers 1 --model_name "distilbert-base-multilingual-cased" --source_len 500 --batch_size 32 --learning_rate 0.001 --epochs 20

echo "cnn + uitnlp/visobert"
python run_train_cls_task.py --task "task-1" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "uitnlp/visobert" --source_len 400 --batch_size 32 --learning_rate 0.001 --epochs 20


echo " ========== TASK 1 EVALUATING =========="
echo "lstm + distilbert-base-multilingual-cased"
python run_evaluation_cls_task.py --task "task-1" --model_type "lstm" --hidden_size 128 --num_layers 1 --model_name "distilbert-base-multilingual-cased" --source_len 500 --batch_size 32

echo "cnn + uitnlp/visobert"
python run_evaluation_cls_task.py --task "task-1" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "uitnlp/visobert" --source_len 400 --batch_size 32


echo " ========== TASK 2 TRAINING =========="
echo "lstm + vinai/phobert-base"
python run_train_cls_task.py --task "task-2" --model_type "lstm" --hidden_size 128 --num_layers 1 --model_name "vinai/phobert-base" --source_len 200 --batch_size 32 --learning_rate 0.001 --epochs 20

echo "cnn + vinai/phobert-base"
python run_train_cls_task.py --task "task-2" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "vinai/phobert-base" --source_len 200 --batch_size 32 --learning_rate 0.001 --epochs 20

echo "lstm + xlm-roberta-base"
python run_train_cls_task.py --task "task-2" --model_type "lstm" --hidden_size 128 --num_layers 1 --model_name "xlm-roberta-base" --source_len 500 --batch_size 32 --learning_rate 0.001 --epochs 20

echo "cnn + uitnlp/visobert"
python run_train_cls_task.py --task "task-2" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "uitnlp/visobert" --source_len 400 --batch_size 32 --learning_rate 0.001 --epochs 20


echo " ========== TASK 2 EVALUATING =========="
echo "lstm + vinai/phobert-base"
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --hidden_size 128 --num_layers 1 --model_name "vinai/phobert-base" --source_len 200 --batch_size 32

echo "cnn + vinai/phobert-base"
python run_evaluation_cls_task.py --task "task-2" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "vinai/phobert-base" --source_len 200 --batch_size 32

echo "lstm + xlm-roberta-base"
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --hidden_size 128 --num_layers 1 --model_name "xlm-roberta-base" --source_len 500 --batch_size 32

echo "cnn + uitnlp/visobert"
python run_evaluation_cls_task.py --task "task-2" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "uitnlp/visobert" --source_len 400 --batch_size 32
