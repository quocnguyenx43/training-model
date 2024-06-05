import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


### For task 1
class SimpleCLSModel(nn.Module):

    def __init__(self, pretrained_model_name, fine_tune=False, num_classes=3):
        super(SimpleCLSModel, self).__init__()

        self.num_classes = num_classes # num classes
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model = AutoModel.from_pretrained(self.pretrained_model_name)
        self.fine_tune = fine_tune

        if self.fine_tune == True:
            self.pretrained_model.train()
            self.pretrained_model.requires_grad_(True)
        else:
            self.pretrained_model.requires_grad_(False)

        # FC
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.pretrained_model.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, input):
        # Pretrained Model
        if self.pretrained_model_name == 'distilbert-base-multilingual-cased':
            model_output = self.pretrained_model(**input).last_hidden_state.mean(dim=1) # Distil
        else:
            model_output = self.pretrained_model(**input).pooler_output # BERT

        # Linear
        fc1_output = F.relu(self.dropout(self.fc1(model_output)))
        fc2_output = F.relu(self.dropout(self.fc2(fc1_output)))

        # Softmax
        soft_max_output = F.log_softmax(self.fc3(fc2_output), dim=1)

        return soft_max_output
    


### For task 2
class SimpleAspectModel(nn.Module):

    def __init__(self, pretrained_model_name, fine_tune=False, num_aspects=4, num_aspect_classes=4):
        super(SimpleAspectModel, self).__init__()

        self.num_aspects = num_aspects # num_aspects
        self.num_aspect_classes = num_aspect_classes # num_aspect_classes
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model = AutoModel.from_pretrained(self.pretrained_model_name)
        self.fine_tune = fine_tune

        if self.fine_tune == True:
            self.pretrained_model.train()
            self.pretrained_model.requires_grad_(True)
        else:
            self.pretrained_model.requires_grad_(False)

        # FCs
        self.fc_layers = nn.ModuleList([
            nn.Linear(self.pretrained_model.config.hidden_size, num_aspect_classes)
            for _ in range(num_aspects)
        ])

        # Dropout layer
        self.dropout_layer = nn.Dropout(p=0.4)

        # Softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        # Pretrained Model
        if self.pretrained_model_name == 'distilbert-base-multilingual-cased':
            model_output = self.pretrained_model(**input).last_hidden_state.mean(dim=1) # Distil
        else:
            model_output = self.pretrained_model(**input).pooler_output # BERT

        # Fully connected layers with Dropout for each aspect
        aspect_outputs = [
            self.dropout_layer(F.relu(fc(model_output)))
            for fc in self.fc_layers
        ]

        # Apply Softmax to each aspect output
        aspect_outputs_softmax = [self.softmax(output) for output in aspect_outputs]
        aspect_outputs_softmax = torch.stack(aspect_outputs_softmax)
        aspect_outputs_softmax = aspect_outputs_softmax.transpose(0, 1)

        return aspect_outputs_softmax


### For task 3