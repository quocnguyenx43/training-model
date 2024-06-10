#!/bin/bash

# Redirect outputs
exec > >(tee -i ./results/logs/train_lai_cnn_complex.log) 2>&1



echo "============================ TASK 1 ============================"
# Training CNN
echo "Training CNN"

echo "cnn + vinai/phobert-base"
python run_train_cls_task.py --task "task-1" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "vinai/phobert-base" --source_len 200 --batch_size 32 --learning_rate 0.001 --epochs 20

echo "cnn + uitnlp/visobert"
python run_train_cls_task.py --task "task-1" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "uitnlp/visobert" --source_len 400 --batch_size 32 --learning_rate 0.001 --epochs 20

echo "cnn + uitnlp/CafeBERT"
python run_train_cls_task.py --task "task-1" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "uitnlp/CafeBERT" --source_len 300 --batch_size 32 --learning_rate 0.001 --epochs 20

echo "cnn + xlm-roberta-base"
python run_train_cls_task.py --task "task-1" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "xlm-roberta-base" --source_len 500 --batch_size 32 --learning_rate 0.001 --epochs 20

echo "cnn + bert-base-multilingual-cased"
python run_train_cls_task.py --task "task-1" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "bert-base-multilingual-cased" --source_len 500 --batch_size 32 --learning_rate 0.001 --epochs 20

echo "cnn + distilbert-base-multilingual-cased"
python run_train_cls_task.py --task "task-1" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "distilbert-base-multilingual-cased" --source_len 500 --batch_size 32 --learning_rate 0.001 --epochs 20


# Evaluating CNN
echo "Evaluating CNN"

echo "cnn + vinai/phobert-base"
python run_evaluation_cls_task.py --task "task-1" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "vinai/phobert-base" --source_len 200 --batch_size 32

echo "cnn + uitnlp/visobert"
python run_evaluation_cls_task.py --task "task-1" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "uitnlp/visobert" --source_len 400 --batch_size 32

echo "cnn + uitnlp/CafeBERT"
python run_evaluation_cls_task.py --task "task-1" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "uitnlp/CafeBERT" --source_len 300 --batch_size 32

echo "cnn + xlm-roberta-base"
python run_evaluation_cls_task.py --task "task-1" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "xlm-roberta-base" --source_len 500 --batch_size 32

echo "cnn + bert-base-multilingual-cased"
python run_evaluation_cls_task.py --task "task-1" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "bert-base-multilingual-cased" --source_len 500 --batch_size 32

echo "cnn + distilbert-base-multilingual-cased"
python run_evaluation_cls_task.py --task "task-1" --model_type "cnn" --num_channels 768 --kernel_size 256 --padding 32 --model_name "distilbert-base-multilingual-cased" --source_len 500 --batch_size 32


