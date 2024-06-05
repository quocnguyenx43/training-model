import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets as dst
import models as md
import functions as func

import argparse as arg

import warnings
warnings.filterwarnings('ignore')

#####################


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
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--fine_tune", action="store_true", default=True)
args = parser.parse_args()

model_type = args.model_type
pretrained_model_name = args.pretrained_model_name
pretrained_model_name_2 = pretrained_model_name.split('/')[-1]
padding_len = args.padding_len
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
fine_tune = args.fine_tune
saving_path = f'./models/task_1/{model_type}_{pretrained_model_name_2}'


### Read data
train_df = pd.read_csv('./data/preprocesseed/train_preprocessed.csv')
dev_df = pd.read_csv('./data/preprocesseed/dev_preprocessed.csv')
test_df = pd.read_csv('./data/preprocesseed/test_preprocessed.csv')


### Dataset & Dataloader
train_dataset = dst.RecruitmentDataset(train_df, tokenizer_name=pretrained_model_name, padding_len=padding_len, task='task-1')
dev_dataset = dst.RecruitmentDataset(dev_df, tokenizer_name=pretrained_model_name, padding_len=padding_len, task='task-1')
test_dataset = dst.RecruitmentDataset(test_df, tokenizer_name=pretrained_model_name, padding_len=padding_len, task='task-1')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


### Model
if model_type == 'simple':
    cls_model = md.SimpleCLSModel(num_classes=3, pretrained_model_name=pretrained_model_name)
elif model_type == 'lstm':
    pass
elif model_type == 'cnn':
    pass


### Training
print(f"Using device: {device}")
print(f'Model type: {model_type}')
print(f'Pretrained model using: {pretrained_model_name}')
print(f'Padding length: {padding_len}')
print(f'Task running: task-1')
print(f'Batch size: {batch_size}')
print(f'Learning rate: {learning_rate}')
print(f'Epochs: {epochs}')
print(f'Do fine tune on pretrained model: {fine_tune}')
print(f'Saving on path: {saving_path}')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cls_model.parameters(), lr=learning_rate)

func.train(
    cls_model, criterion, optimizer,
    epochs=epochs,
    train_dataloader=train_dataloader, dev_dataloader=dev_dataloader,
    saving_path=saving_path,
    task_running='task-1',
)