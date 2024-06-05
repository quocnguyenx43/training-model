import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import datasets as dst
import models as md
import functions_train_task_3 as func

import argparse as arg

from rich.console import Console

import warnings
warnings.filterwarnings('ignore')

#####################

console = Console(record=True)


try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except RuntimeError as e:
    device = 'cpu'


### Params
parser = arg.ArgumentParser(description="Params")
parser.add_argument("--model_name", type=str, default='VietAI/vit5-base')
parser.add_argument("--source_len", type=int, default=512)
parser.add_argument("--target_len", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

model_name = args.model_name
source_len = args.source_len
target_len = args.target_len
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
saving_path = f'./models/task_3/{model_name}'

print(source_len)


### Read data
train_df = pd.read_csv('./data/small/train_preprocessed.csv')
dev_df = pd.read_csv('./data/small/dev_preprocessed.csv')
test_df = pd.read_csv('./data/small/test_preprocessed.csv')


### Dataset & Dataloader
train_dataset = dst.RecruitmentDataset(train_df, tokenizer_name=model_name, padding_len=source_len, target_len=target_len, task='task-3')
dev_dataset = dst.RecruitmentDataset(dev_df, tokenizer_name=model_name, padding_len=source_len, target_len=target_len, task='task-3')
# test_dataset = dst.RecruitmentDataset(test_df, tokenizer_name=model_name, padding_len=source_len, target_len=target_len, task='task-3')

#
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


### Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
generation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


### Training
print(f"Using device: {device}")
print(f'Model name: {model_name}')
print(f'Source len: {source_len}')
print(f'Target len: {target_len}')
print(f'Task running: task-3')
print(f'Batch size: {batch_size}')
print(f'Learning rate: {learning_rate}')
print(f'Epochs: {epochs}')
print(f'Saving on path: {saving_path}')


optimizer = optim.Adam(generation_model.parameters(), lr=learning_rate)
func.train(
    generation_model, optimizer, tokenizer,
    epochs=epochs,
    train_dataloader=train_dataloader,
    saving_path=saving_path,
    device=device
)