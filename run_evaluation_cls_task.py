import os

import torch
import torch.nn as nn

import models as md
import functions as func
import my_import as imp


### Find weight model path (lastest epoch)
task_folder = imp.args['task'].replace('-', '_')
prefix_file_name = imp.args['model_type'] + '_' + imp.args['model_name'].split('/')[-1]
all_model_files = os.listdir('./models/' + task_folder + '/')

lastest = 0
for file in all_model_files:
    if file.startswith(prefix_file_name):
        a = int(file.split('_')[-1].split('.')[0])
        if a > lastest:
            lastest = a

lastest = str(lastest)
model_weight_path = './models/' + task_folder + '/' + prefix_file_name + '_' + lastest + '.pth'


###### Model
# if simple
if imp.args['model_type'] == 'simple':
    if imp.args['task'] == 'task-1':
        model = md.SimpleCLSModel(pretrained_model_name=imp.args['model_name'])
    elif imp.args['task'] == 'task-2':
        model = md.SimpleAspectModel(pretrained_model_name=imp.args['model_name'])
# if lstm
elif imp.args['model_type'] == 'lstm':
    if imp.args['task'] == 'task-1':
        model = md.ComplexCLSModel(
            model_type='lstm',
            params=imp.params,
            pretrained_model_name=imp.args['model_name'],
        )
    elif imp.args['task'] == 'task-2':
        model = md.ComplexAspectModel(
            model_type='lstm',
            params=imp.params,
            pretrained_model_name=imp.args['model_name'],
        )
# if cnn
elif imp.args['model_type'] == 'cnn':
    if imp.args['task'] == 'task-1':
        model = md.ComplexCLSModel(
            model_type='cnn',
            params=imp.params,
            pretrained_model_name=imp.args['model_name'],
        )
    elif imp.args['task'] == 'task-2':
        model = md.ComplexAspectModel(
            model_type='cnn',
            params=imp.params,
            pretrained_model_name=imp.args['model_name'],
        )
    

###### Loading weights
model = model.to(imp.device)
model.load_state_dict(torch.load(model_weight_path))
criterion = nn.CrossEntropyLoss()
print(f'model_weight_path: {model_weight_path}')
print('Loading model weight successfully!\n')


### Evaluating on Dev set
print('Evaluation on dev test')
func.evaluate(
    model, criterion,
    dataloader=imp.dev_dataloader,
    task_running=imp.args['task'],
    cm=True, cr=True, last_epoch=True,
    device=imp.args['device'],
)


### Evaluating on Test set
print('Evaluation on test test')
func.evaluate(
    model, criterion,
    dataloader=imp.test_dataloader, 
    task_running=imp.args['task'],
    cm=True, cr=True, last_epoch=True,
    device=imp.args['device'],
)