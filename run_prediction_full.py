import pandas as pd

import torch
from torch.utils.data import DataLoader

import argparse as arg
from rich.console import Console
import warnings

import my_datasets as dst

import os

import torch
import torch.nn as nn

import models as md
import functions as func


#####################
console = Console(record=True)
warnings.filterwarnings("ignore")


parser = arg.ArgumentParser(description="Params")

parser.add_argument("--source_len_1", type=int, default=200)
parser.add_argument("--source_len_2", type=int, default=200)
parser.add_argument("--source_len_3", type=int, default=1024)
parser.add_argument("--target_len", type=int, default=200)

parser.add_argument("--batch_size", type=int, default=12)

# for LSTM
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=1)

# for CNN
parser.add_argument("--num_channels", type=int, default=768)
parser.add_argument("--kernel_size", type=int, default=256)
parser.add_argument("--padding", type=int, default=32)

# HA
parser.add_argument("--path1", type=str, default="")
parser.add_argument("--path2", type=str, default="")
parser.add_argument("--path3", type=str, default="")

args = parser.parse_args()
args = vars(args)

task_1_model_path = args['path1']
task_2_model_path = args['path2']
task_3_model_path = args['path3']

padding_1 = args['source_len_1']
padding_2 = args['source_len_2']
padding_3 = args['source_len_3']

target_len = args['target_len']

batch_size = args['batch_size']

params = {
    'hidden_size': args['hidden_size'],
    'num_layers': args['num_layers'],
    'num_channels': args['num_channels'],
    'kernel_size': args['kernel_size'],
    'padding': args['padding'],
}


print(f'path1: {task_1_model_path}')
print(f'path2: {task_2_model_path}')
print(f'path3: {task_3_model_path}')
print()



### TASK 1
print('task 1')
test_df = pd.read_csv('./data/preprocessed/test_preprocessed_old.csv').head(10)
args['test_shape'] = test_df.shape
test_dataset = dst.RecruitmentDataset(
    test_df, tokenizer_name='uitnlp/visobert',
    padding_len=padding_1, target_len=128,
    task='task-1',
)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model1 = md.ComplexCLSModel(
    model_type='cnn',
    params=params,
    pretrained_model_name='uitnlp/visobert',
)

model1 = model1.to('cuda')
model1.load_state_dict(torch.load(task_1_model_path))
criterion = nn.CrossEntropyLoss()
print(f'model_weight_path: {task_1_model_path}')
print('Loading model weight successfully!\n')

print('Task 1 prediction')
predictions_1, _, _ = func.evaluate(
    model1, criterion,
    dataloader=test_dataloader, 
    task_running='task-1',
    cm=True, cr=True, last_epoch=True,
    device='cuda',
)


## TASK 2
print('task 2')
test_df = pd.read_csv('./data/preprocessed/test_preprocessed_old.csv').head(10)
args['test_shape'] = test_df.shape
test_dataset = dst.RecruitmentDataset(
    test_df, tokenizer_name='vinai/phobert-base',
    padding_len=padding_2, target_len=128,
    task='task-2',
)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model2 = md.ComplexAspectModel(
    model_type='lstm',
    params=params,
    pretrained_model_name='vinai/phobert-base',
)

print(f'model_weight_path: {task_2_model_path}')
model2 = model2.to('cuda')
model2.load_state_dict(torch.load(task_2_model_path))
criterion = nn.CrossEntropyLoss()
print('Loading model weight successfully!\n')

print('Task 2 prediction')
predictions_2, _, _ = func.evaluate(
    model2, criterion,
    dataloader=test_dataloader, 
    task_running='task-2',
    cm=True, cr=True, last_epoch=True,
    device='cuda',
)


mapping_aspect = {
    0      : 'trung tính',
    1      : 'tích cực',
    2      : 'tiêu cực',
    3      : 'không đề cập',
}

mapping_label = {
    0         : 'rõ ràng',
    1         : 'cảnh báo',
    2         : 'có yếu tố thu hút',
}


df1 = pd.DataFrame(predictions_1, columns=['predicted_label'])
df2 = pd.DataFrame(predictions_2, columns=['predicted_title', 'predicted_desc', 'predicted_comp', 'predicted_other'])
df_merged = pd.concat([df1, df2], axis=1)
df_merged.to_csv('results/' + task_1_model_path.replace('.', '_').replace('/', '_') + task_2_model_path.replace('.', '_').replace('/', '_') + '.csv')
df_merged.predicted_label.map(mapping_label)
df_merged.predicted_title.map(mapping_aspect)
df_merged.predicted_desc.map(mapping_aspect)
df_merged.predicted_comp.map(mapping_aspect)
df_merged.predicted_other.map(mapping_aspect)
print(df_merged)

def adding_previous_tasks(df):
    previous_task_outputs = []
    for index in range(len(df)): 
        s = "khía cạnh tiêu đề: " + df.iloc[index]['predicted_title'] + " [SEP] " \
            + "khía cạnh mô tả: " + df.iloc[index]['predicted_desc'] + " [SEP] " \
            + "khía cạnh công ty: " + df.iloc[index]['predicted_comp'] + " [SEP] " \
            + "khía cạnh khác: " + df.iloc[index]['predicted_other'] + " [SEP] " \
            + "nhãn chung: " + df.iloc[index]['predicted_label']  + " [SEP] "
        previous_task_outputs.append(s[:-1])

    df['pre_tasks'] = previous_task_outputs

df_merged = adding_previous_tasks(df_merged)
print(df_merged)