import random
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

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



#####################
console = Console(record=True)
warnings.filterwarnings("ignore")

parser = arg.ArgumentParser(description="Params")

parser.add_argument("--path1", type=str, default="")
parser.add_argument("--path2", type=str, default="")
parser.add_argument("--path3", type=str, default="")

parser.add_argument("--source_len_1", type=int, default=200)
parser.add_argument("--source_len_2", type=int, default=200)
parser.add_argument("--source_len_3", type=int, default=768)
parser.add_argument("--target_len", type=int, default=128)

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--device", type=str, default='cuda')

# for LSTM
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=1)

# for CNN
parser.add_argument("--num_channels", type=int, default=768)
parser.add_argument("--kernel_size", type=int, default=256)
parser.add_argument("--padding", type=int, default=32)

args = parser.parse_args()
args = vars(args)

model_path_1 = args['path1']
model_path_2 = args['path2']
model_path_3 = args['path3']

model_name_mapping = {
    'phobert-base': 'vinai/phobert-base',
    'visobert': 'uitnlp/visobert',
    'CafeBERT': 'uitnlp/CafeBERT',
    'xlm-roberta-base': 'xlm-roberta-base',
    'bert-base-multilingual-cased': 'bert-base-multilingual-cased',
    'distilbert-base-multilingual-cased': 'distilbert-base-multilingual-cased',
    'vit5-base': 'VietAI/vit5-base',
    'bartpho-syllable-base': 'vinai/bartpho-syllable-base',
    'bartpho-word-base': 'vinai/bartpho-word-base',
}

model_name_1 = model_name_mapping[model_path_1.split('_')[1]]
model_name_2 = model_name_mapping[model_path_2.split('_')[1]]
model_name_3 = model_name_mapping[model_path_3.split('_')[0]]

model_type_1 = model_path_1.split('_')[0]
model_type_2 = model_path_2.split('_')[0]
model_type_3 = 'simple'

padding_1 = args['source_len_1']
padding_2 = args['source_len_2']
padding_3 = args['source_len_3']
target_padding = args['target_len']

model_path_1 = './models/task_1/' + model_path_1
model_path_2 = './models/task_2/' + model_path_2
model_path_3 = './models/task_3/' + model_path_3

batch_size = args['batch_size']
device = args['device']
params = {
    'hidden_size': args['hidden_size'],
    'num_layers': args['num_layers'],
    'num_channels': args['num_channels'],
    'kernel_size': args['kernel_size'],
    'padding': args['padding'],
}

print(f'model_1: {model_type_1}, model_name_1: {model_name_1}, path_1: {model_path_1}')
print(f'model_2: {model_type_2}, model_name_2: {model_name_2}, path_2: {model_path_2}')
print(f'model_3: {model_type_3}, model_name_3: {model_name_3}, path_3: {model_path_3}')
print()

test_df = pd.read_csv('./data/preprocessed/test_preprocessed.csv')
test_df.explanation = test_df.explanation.fillna('')
print(f'test shape: {test_df.shape}')
print()


# Create dataloader
def create_dataloader(df, model_name, source_padding, target_padding, task='task-1'):
    test_dataset = dst.RecruitmentDataset(
        df, tokenizer_name=model_name,
        padding_len=source_padding, target_len=target_padding,
        task=task,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataloader

# Load model
def load_model(task, model_type, model_name, model_path, params):
    # if task_1
    if task == 'task-1':
        if model_type == 'simple':
            model = md.SimpleCLSModel(pretrained_model_name=model_name)
        else:
            model = md.ComplexCLSModel(model_type=model_type, params=params, pretrained_model_name=model_name)
    
    # if task_2
    elif task == 'task-2':
        if model_type == 'simple':
            model = md.SimpleAspectModel(pretrained_model_name=model_name)
        else:
            model = md.ComplexAspectModel(model_type=model_type, params=params, pretrained_model_name=model_name)

    elif task == 'task-3':
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model = model.to(device)
    print(f'model_weight_path: {model_path}')
    model.load_state_dict(torch.load(model_path))
    print('Loading model weight successfully!\n')
        
    return model


### TASK 1
print('TASK 1')
task_1_dataloader = create_dataloader(test_df, model_name_1, padding_1, None, 'task-1')
model_1 = load_model('task-1', model_type_1, model_name_1, model_path_1, params)
criterion = nn.CrossEntropyLoss()

print('TASK 1 PREDICTION')
predictions_task_1, _, _ = func.evaluate(
    model_1, criterion,
    dataloader=task_1_dataloader, 
    task_running='task-1',
    cm=True, cr=True, last_epoch=True,
    device=device,
)


### TASK 2
print('TASK 2')
task_2_dataloader = create_dataloader(test_df, model_name_2, padding_2, None, 'task-2')
model_2 = load_model('task-2', model_type_2, model_name_2, model_path_2, params)
criterion = nn.CrossEntropyLoss()

print('TASK 2 PREDICTION')
predictions_task_2, _, _ = func.evaluate(
    model_2, criterion,
    dataloader=task_2_dataloader, 
    task_running='task-2',
    cm=True, cr=True, last_epoch=True,
    device=device,
)

### TASK 3
mapping_aspect = {0: 'trung tính', 1: 'tích cực', 2: 'tiêu cực', 3: 'không đề cập'}
mapping_label = {0: 'rõ ràng', 1: 'cảnh báo', 2: 'có yếu tố thu hút'}

df1 = pd.DataFrame(predictions_task_1, columns=['predicted_label'])
df2 = pd.DataFrame(predictions_task_2, 
                   columns=['predicted_title', 'predicted_desc', 'predicted_comp', 'predicted_other'])
df_predictions = pd.concat([df1, df2], axis=1)

def adding_previous_tasks(df):
    previous_task_outputs = []
    for index in range(len(df)): 
        s = "khía cạnh tiêu đề: " + mapping_aspect[df.iloc[index]['predicted_title']] + " [SEP] " \
            + "khía cạnh mô tả: " + mapping_aspect[df.iloc[index]['predicted_desc']] + " [SEP] " \
            + "khía cạnh công ty: " + mapping_aspect[df.iloc[index]['predicted_comp']] + " [SEP] " \
            + "khía cạnh khác: " + mapping_aspect[df.iloc[index]['predicted_other']] + " [SEP] " \
            + "nhãn chung: " + mapping_label[df.iloc[index]['predicted_label']]  + " [SEP] "
        
        previous_task_outputs.append(s[:-1])

    df['pre_tasks'] = previous_task_outputs
    return df

df_predictions = adding_previous_tasks(df_predictions)
test_df.pre_tasks = df_predictions.pre_tasks

print('TASK 3')
task_3_dataloader = create_dataloader(test_df, model_name_3, padding_3, target_padding, 'task-3')
model_3 = load_model('task-3', model_type_3, model_name_3, model_path_3, params)
tokenizer_3 = AutoTokenizer.from_pretrained(model_name_3)

print('TASK 3 PREDICTION')
predictions_3, _ = func.generate_task_3(
    model_3, tokenizer_3,
    task_3_dataloader, target_len=target_padding,
    device=device
)

df_predictions['generated_text'] = pd.Series(predictions_3)

saving_path = 'outputs/' + \
               model_name_1.replace('/', '-') + \
               model_name_2.replace('/', '-') + \
               model_name_3.replace('/', '-') + '.csv'
print(f'saving_path: {saving_path}')

df_predictions.to_csv(saving_path)