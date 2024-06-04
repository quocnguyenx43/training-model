import pandas as pd
import numpy as np

from transformers import AutoTokenizer
from transformers import AutoModel

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import mdatasets
import models
import function_task_1

import argparse

parser = argparse.ArgumentParser(description="Params")
parser.add_argument("--model_name", type=str, default="vinai/phobert-base",)
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--padding_len", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--fine_tune", action="store_true", default=True)
args = parser.parse_args()

### Hypers
device = 'cpu'
model_name = args.model_name
num_classes = args.num_classes
padding_len = args.padding_len
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
fine_tune = args.fine_tune
saving_path = f'models/task_1/{model_name}.pth' # change to some path in a string

### Read data
train_df = pd.read_csv('./data/small/train_preprocessed.csv')
dev_df = pd.read_csv('./data/small/dev_preprocessed.csv')
test_df = pd.read_csv('./data/small/test_preprocessed.csv')


### Model
pretrained_model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
cls_model = models.CLSModel(num_classes=3, pretrained_model=pretrained_model)

# Dataset & Dataloader
train_dataset = mdatasets.RecruitmentDataset(train_df, tokenizer, padding_length=padding_len, type='task_1')
dev_dataset = mdatasets.RecruitmentDataset(dev_df, tokenizer, padding_length=padding_len, type='task_1')
test_dataset = mdatasets.RecruitmentDataset(test_df, tokenizer, padding_length=padding_len, type='task_1')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cls_model.parameters(), lr=learning_rate)

function_task_1.train(
    cls_model, criterion, optimizer, epochs=3,
    train_dataloader=train_dataloader, 
    dev_dataloader=dev_dataloader,
    device=device,
    saving_path=saving_path,
)

torch.save(cls_model.state_dict(), saving_path)