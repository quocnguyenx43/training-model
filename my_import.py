import pandas as pd

import torch
from torch.utils.data import DataLoader

import argparse as arg
from rich.console import Console

import datasets as dst


#####################
console = Console(record=True)



### Params 
parser = arg.ArgumentParser(description="Params")
parser.add_argument("--task", type=str, default='task-1')
parser.add_argument("--model_type", type=str, default='simple')
parser.add_argument("--model_name", type=str, default='vinai/phobert-base')
parser.add_argument("--source_len", type=int, default=200)
parser.add_argument("--target_len", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--fine_tune", action="store_true", default=True)
args = parser.parse_args()
args = vars(args)

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except RuntimeError as e:
    device = 'cpu'
saving_path = './models/task_1/' + args['model_type'] + '/' + args['model_name'].split('/')[-1]

args['device'] = device
args['saving_path'] = saving_path


### Read data
train_df = pd.read_csv('./data/small/train_preprocessed.csv')
dev_df = pd.read_csv('./data/small/dev_preprocessed.csv')
test_df = pd.read_csv('./data/small/test_preprocessed.csv')


### Dataset
train_dataset = dst.RecruitmentDataset(
    train_df, tokenizer_name=args['model_name'],
    padding_len=args['source_len'], target_len=args['target_len'],
    task=args['task'],
)
dev_dataset = dst.RecruitmentDataset(
    dev_df, tokenizer_name=args['model_name'],
    padding_len=args['source_len'], target_len=args['target_len'],
    task=args['task'],
)
test_dataset = dst.RecruitmentDataset(
    test_df, tokenizer_name=args['model_name'],
    padding_len=args['source_len'], target_len=args['target_len'],
    task=args['task'],
)


### Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=args['batch_size'], shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)


### Printing args
for key, value in vars(args).items():
    if args['task'] == 'task-1' and key == 'target_len':
        continue
    if args['task'] == 'task-3' and key == 'model_type' or key == 'fine_tune':
        continue
    console.log(f'{key}: {value}')