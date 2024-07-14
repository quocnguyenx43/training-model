#!/bin/bash

# Redirect outputs
exec > >(tee -i ./results/logs_raw/task_2_lstm_cnn_eval_log.log) 2>&1


# Evaluating LSTM
echo "Evaluating LSTM"

echo "lstm + vinai/phobert-base"
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --hidden_size 128 --num_layers 1 --model_name "vinai/phobert-base" --source_len 256 --batch_size 32

echo "lstm + uitnlp/visobert"
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --hidden_size 128 --num_layers 1 --model_name "uitnlp/visobert" --source_len 512 --batch_size 32

echo "lstm + uitnlp/CafeBERT"
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --hidden_size 128 --num_layers 1 --model_name "uitnlp/CafeBERT" --source_len 512 --batch_size 32

echo "lstm + xlm-roberta-base"
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --hidden_size 128 --num_layers 1 --model_name "xlm-roberta-base" --source_len 512 --batch_size 32

echo "lstm + bert-base-multilingual-cased"
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --hidden_size 128 --num_layers 1 --model_name "bert-base-multilingual-cased" --source_len 512 --batch_size 32

echo "lstm + distilbert-base-multilingual-cased"
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --hidden_size 128 --num_layers 1 --model_name "distilbert-base-multilingual-cased" --source_len 512 --batch_size 32


# Evaluating CNN
echo "Evaluating CNN"

echo "cnn + vinai/phobert-base"
python run_evaluation_cls_task.py --task "task-2" --model_type "cnn" --num_channels 64 --kernel_size 64 --padding 64 --model_name "vinai/phobert-base" --source_len 256 --batch_size 32

echo "cnn + uitnlp/visobert"
python run_evaluation_cls_task.py --task "task-2" --model_type "cnn" --num_channels 64 --kernel_size 64 --padding 64 --model_name "uitnlp/visobert" --source_len 512 --batch_size 32

echo "cnn + uitnlp/CafeBERT"
python run_evaluation_cls_task.py --task "task-2" --model_type "cnn" --num_channels 64 --kernel_size 64 --padding 64 --model_name "uitnlp/CafeBERT" --source_len 512 --batch_size 32

echo "cnn + xlm-roberta-base"
python run_evaluation_cls_task.py --task "task-2" --model_type "cnn" --num_channels 64 --kernel_size 64 --padding 64 --model_name "xlm-roberta-base" --source_len 512 --batch_size 32

echo "cnn + bert-base-multilingual-cased"
python run_evaluation_cls_task.py --task "task-2" --model_type "cnn" --num_channels 64 --kernel_size 64 --padding 64 --model_name "bert-base-multilingual-cased" --source_len 512 --batch_size 32

echo "cnn + distilbert-base-multilingual-cased"
python run_evaluation_cls_task.py --task "task-2" --model_type "cnn" --num_channels 64 --kernel_size 64 --padding 64 --model_name "distilbert-base-multilingual-cased" --source_len 512 --batch_size 32