import torch
import torch.nn as nn
import torch.optim as optim

import models as md
import functions as func
import my_import as imp



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
        pass
# if cnn
elif imp.args['model_type'] == 'cnn':
    if imp.args['task'] == 'task-1':
        model = md.ComplexCLSModel(
            model_type='cnn',
            params=imp.params,
            pretrained_model_name=imp.args['model_name'],
        )
    elif imp.args['task'] == 'task-2':
        pass
    

###### Change device & setting criterion, optimizer
model = model.to(imp.args['device'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=imp.args['learning_rate'])


###### Training
print('Training ...')
func.train(
    model, criterion, optimizer,
    epochs=imp.args['epochs'],
    train_dataloader=imp.train_dataloader, dev_dataloader=imp.dev_dataloader,
    saving_path=imp.args['saving_path'],
    task_running=imp.args['task'],
    device=imp.args['device']
)
