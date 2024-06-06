import torch
import torch.nn as nn
import torch.optim as optim

import models as md
import functions as func
import my_import as imp



###### Model
if imp.args['model_type'] == 'simple':
    cls_model = md.SimpleCLSModel(num_classes=3, pretrained_model_name=imp.args['model_name'])
elif imp.args['model_type'] == 'lstm':
    pass
elif imp.args['model_type'] == 'cnn':
    pass


###### Change device & setting criterion, optimizer
cls_model = cls_model.to(imp.args['device'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cls_model.parameters(), lr=imp.args['learning_rate'])


###### Training
print('Training ...')
func.train(
    cls_model, criterion, optimizer,
    epochs=imp.args['epochs'],
    train_dataloader=imp.train_dataloader, dev_dataloader=imp.dev_dataloader,
    saving_path=imp.args['saving_path'],
    task_running=imp.args['task'],
    device=imp.args['device']
)
