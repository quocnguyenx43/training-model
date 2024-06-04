import torch
import torch.nn as nn
import torch.nn.functional as F

class CLSModel(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(CLSModel, self).__init__()

        self.num_classes = num_classes # num classes
        self.pretrained_model = pretrained_model # pretrained model

        # FC
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.pretrained_model.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, input):

        # Pretrained Model
        if self.pretrained_model == 'distilbert-base-multilingual-cased':
            model_output = self.pretrained_model(**input).last_hidden_state.mean(dim=1) # Distil
        else:
            model_output = self.pretrained_model(**input).pooler_output # BERT

        # Linear
        fc1_output = F.relu(self.dropout(self.fc1(model_output)))
        fc2_output = F.relu(self.dropout(self.fc2(fc1_output)))

        # Softmax output for classification
        soft_max_output = F.log_softmax(self.fc3(fc2_output), dim=1)

        return soft_max_output