import os

import torch
import torch.nn as nn

import models as md
import functions as func
import my_import as imp


### Find weight model path
all_model_files = os.listdir(f'./models/{imp.args['task']}/')
lastest = 0
for file in all_model_files:
    if file.startswith(f'{imp.args['model_type']}_{imp.args['model_name']}'):
        a = int(file.split('_')[-1].split('.')[0])
        if a > lastest:
            lastest = a
model_weight_path = f'./models/{imp.args['task']}/{imp.args['model_type']}_{imp.args['model_name']}_{lastest}.pth'


###### Model
if imp.args['model_type'] == 'simple':
    cls_model = md.SimpleCLSModel(num_classes=3, pretrained_model_name=imp.args['model_name'])
elif imp.args['model_type'] == 'lstm':
    pass
elif imp.args['model_type'] == 'cnn':
    pass


###### Loading weights
cls_model = cls_model.to(imp.device)
cls_model.load_state_dict(torch.load(model_weight_path))
criterion = nn.CrossEntropyLoss()
imp.console.log(f'model_weight_path: {model_weight_path}')
imp.console.log('Loading model weight successfully!\n')


### Evaluating on Dev set
imp.console.log('Evaluation on dev test')
func.evaluate(
    cls_model, criterion,
    dataloader=imp.dev_dataloader, 
    task_running=imp.args['task'],
    cm=True, cr=True, last_epoch=True,
    device=imp.args['device'],
)


### Evaluating on Test set
imp.console.log('Evaluation on test test')
func.evaluate(
    cls_model, criterion,
    dataloader=imp.test_dataloader, 
    task_running=imp.args['task'],
    cm=True, cr=True, last_epoch=True,
    device=imp.args['device'],
)