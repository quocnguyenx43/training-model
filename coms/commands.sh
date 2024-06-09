#!/bin/bash

# Redirect outputs
exec > >(tee -i ./results/logs/test_log.log) 2>&1

# cafebert
python run_train_cls_task.py --task "task-1" --model_type "simple" --model_name "uitnlp/CafeBERT" --source_len 400 --batch_size 32 --learning_rate 0.01 --epochs 20
python run_evaluation_cls_task.py --task "task-1" --model_type "simple" --model_name "uitnlp/CafeBERT" --source_len 400 --batch_size 32

rm -rf models/task_1
mkdir models/task_1

python run_train_cls_task.py --task "task-1" --model_type "simple" --model_name "uitnlp/CafeBERT" --source_len 300 --batch_size 32 --learning_rate 0.01 --epochs 20
python run_evaluation_cls_task.py --task "task-1" --model_type "simple" --model_name "uitnlp/CafeBERT" --source_len 300 --batch_size 32

rm -rf models/task_1
mkdir models/task_1

python run_train_cls_task.py --task "task-1" --model_type "simple" --model_name "uitnlp/CafeBERT" --source_len 200 --batch_size 32 --learning_rate 0.01 --epochs 20
python run_evaluation_cls_task.py --task "task-1" --model_type "simple" --model_name "uitnlp/CafeBERT" --source_len 200 --batch_size 32

rm -rf models/task_1
mkdir models/task_1

# xlm
python run_train_cls_task.py --task "task-1" --model_type "simple" --model_name "xlm-roberta-base" --source_len 500 --batch_size 32 --learning_rate 0.01 --epochs 20
python run_evaluation_cls_task.py --task "task-1" --model_type "simple" --model_name "xlm-roberta-base" --source_len 500 --batch_size 32

rm -rf models/task_1
mkdir models/task_1

python run_train_cls_task.py --task "task-1" --model_type "simple" --model_name "xlm-roberta-base" --source_len 400 --batch_size 32 --learning_rate 0.01 --epochs 20
python run_evaluation_cls_task.py --task "task-1" --model_type "simple" --model_name "xlm-roberta-base" --source_len 400 --batch_size 32

rm -rf models/task_1
mkdir models/task_1

python run_train_cls_task.py --task "task-1" --model_type "simple" --model_name "xlm-roberta-base" --source_len 300 --batch_size 32 --learning_rate 0.01 --epochs 20
python run_evaluation_cls_task.py --task "task-1" --model_type "simple" --model_name "xlm-roberta-base" --source_len 300 --batch_size 32

rm -rf models/task_1
mkdir models/task_1

# bertbase
python run_train_cls_task.py --task "task-1" --model_type "simple" --model_name "bert-base-multilingual-cased" --source_len 500 --batch_size 32 --learning_rate 0.01 --epochs 20
python run_evaluation_cls_task.py --task "task-1" --model_type "simple" --model_name "bert-base-multilingual-cased" --source_len 500 --batch_size 32

rm -rf models/task_1
mkdir models/task_1

python run_train_cls_task.py --task "task-1" --model_type "simple" --model_name "bert-base-multilingual-cased" --source_len 400 --batch_size 32 --learning_rate 0.01 --epochs 20
python run_evaluation_cls_task.py --task "task-1" --model_type "simple" --model_name "bert-base-multilingual-cased" --source_len 400 --batch_size 32

rm -rf models/task_1
mkdir models/task_1

python run_train_cls_task.py --task "task-1" --model_type "simple" --model_name "bert-base-multilingual-cased" --source_len 300 --batch_size 32 --learning_rate 0.01 --epochs 20
python run_evaluation_cls_task.py --task "task-1" --model_type "simple" --model_name "bert-base-multilingual-cased" --source_len 300 --batch_size 32

rm -rf models/task_1
mkdir models/task_1