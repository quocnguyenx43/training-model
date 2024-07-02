import os

import torch
import torch.nn as nn

import models as md
import functions as func
import my_import as imp


task_1_model_path = imp.args['path1']
task_2_model_path = imp.args['path2']
task_3_model_path = imp.args['path3']

print(f'path1: {task_1_model_path}')
print(f'path2: {task_2_model_path}')
print(f'path3: {task_3_model_path}')

params = {
    'hidden_size': 128,
    'num_layers': 1,
    'num_channels': 768,
    'kernel_size': 256,
    'padding': 32
}

# Task 1
model1 = md.ComplexCLSModel(
    model_type='cnn',
    params=params,
    pretrained_model_name='uitnlp/visobert',
)

model1 = model1.to(imp.device)
model1.load_state_dict(torch.load(task_1_model_path))
criterion = nn.CrossEntropyLoss()
print(f'model_weight_path: {task_1_model_path}')
print('Loading model weight successfully!\n')

print('Task 1 prediction')
predictions, true_labels, _ = func.evaluate(
    model1, criterion,
    dataloader=imp.test_dataloader, 
    task_running='task-1',
    cm=True, cr=True, last_epoch=True,
    device='cuda',
)

print(predictions, true_labels)


# # Task 2
# model2 = md.ComplexAspectModel(
#     model_type='lstm',
#     params=params,
#     pretrained_model_name='vinai/phobert-base',
# )
