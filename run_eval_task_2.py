import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets as dst
import models as md
import functions as func

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
parser.add_argument("--model_type", type=str, default='simple')
parser.add_argument("--pretrained_model_name", type=str, default='vinai/phobert-base')
parser.add_argument("--padding_len", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=12)
args = parser.parse_args()

model_type = args.model_type
pretrained_model_name = args.pretrained_model_name
pretrained_model_name_2 = pretrained_model_name.split('/')[-1]
padding_len = args.padding_len
batch_size = args.batch_size

### Find weight model path
all_model_files = os.listdir('./models/task_2/')
lastest = 0
for file in all_model_files:
    if file.startswith(f'{model_type}_{pretrained_model_name_2}'):
        a = int(file.split('_')[-1].split('.')[0])
        if a > lastest:
            lastest = a
model_weight_path = f'./models/task_2/{model_type}_{pretrained_model_name_2}_{lastest}.pth'


### Read data
dev_df = pd.read_csv('./data/preprocessed/dev_preprocessed.csv')
test_df = pd.read_csv('./data/preprocessed/test_preprocessed.csv')

### Dataset & Dataloader
dev_dataset = dst.RecruitmentDataset(dev_df, tokenizer_name=pretrained_model_name, padding_len=padding_len, task='task-2')
test_dataset = dst.RecruitmentDataset(test_df, tokenizer_name=pretrained_model_name, padding_len=padding_len, task='task-2')

dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


### Model
if model_type == 'simple':
    aspect_model = md.SimpleAspectModel(pretrained_model_name=pretrained_model_name, fine_tune=True).to(device)
elif model_type == 'lstm':
    pass
elif model_type == 'cnn':
    pass


### Print params
console.log(f"Using device: {device}")
console.log(f'Model type: {model_type}')
console.log(f'Pretrained model using: {pretrained_model_name}')
console.log(f'Padding length: {padding_len}')
console.log(f'Task running: task-2')
console.log(f'Batch size: {batch_size}')
console.log(f'Model path: {model_weight_path}')


### Loading weights
aspect_model.load_state_dict(torch.load(model_weight_path))
criterion = nn.CrossEntropyLoss()
console.log('Loading model weights successfully!\n')

### Evaluating on Dev set
print('Evaluation on dev test')
func.evaluate(
    aspect_model, criterion,
    dataloader=dev_dataloader, 
    task_running='task-2',
    cm=True, cr=True, last_epoch=True,
    device=device
)

### Evaluating on Test set
print('Evaluation on test test')
func.evaluate(
    aspect_model, criterion,
    dataloader=dev_dataloader, 
    task_running='task-2',
    cm=True, cr=True, last_epoch=True,
    device=device
)