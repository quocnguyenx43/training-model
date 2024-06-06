import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import utils

    

class RecruitmentDataset(Dataset):

    def __init__(self, df, tokenizer_name, padding_len, task='task-1', target_len=None):
        self.df = df

        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.padding_len = padding_len
        self.target_len = target_len
        self.task = task

        self.data_x = self.create_X()
        self.data_y = self.create_y()


    def __len__(self):
        return len(self.df)


    def create_X(self):
        infor_cols = [
            'tiêu đề', 'ngành nghề',
            'mô tả', 'kinh nghiệm', 'học vấn', 'bằng cấp', 'quyền lợi',
            'tên công ty', 'địa chỉ', 'số điện thoại', 'người liên hệ',
            # 'tên người đăng', 'số điện thoại người đăng', 'ngày đăng', 'hạn nộp CV', 'tin đăng ẩn danh', 'là nhà tuyển dụng',
            'số lượng tuyển', 'hình thức hợp đồng', 'hình thức trả lương', 'lương tối thiểu', 'lương tối đa', 'giới tính', 'năm sinh', 'tuổi', 'tuổi thấp nhất', 'tuổi cao nhất',
        ]

        # Get data
        pre_tasks = self.df['pre_tasks']
        X_data = self.df.drop(['title_aspect', 'desc_aspect', 'company_aspect', 'other_aspect', 'label', 'explanation', 'pre_tasks'], axis=1)

        # Combine
        X_combined = []
        for q, row in X_data.iterrows():
            inp = '[CLS] '
            for idx, content in enumerate(row):
                inp = inp + infor_cols[idx] + ": " + str(content) + " [SEP] "
            if self.task == 'task-3':
                inp = inp + pre_tasks[q]
            else:
                inp = inp[:-1]
            X_combined.append(inp)

        return X_combined

    def create_y(self):
        if self.task == 'task-1':
            label = torch.tensor(self.df['label'])
            label = F.one_hot(label, num_classes=3).float()
        elif self.task =='task-2':
            label = self.df[['title_aspect', 'desc_aspect', 'company_aspect', 'other_aspect']].to_numpy()
            label = np.eye(4)[label]
            label = torch.from_numpy(label).float()
        else:
            label = self.df['explanation'].to_numpy()

        return list(label)


    def __getitem__(self, index):
        item = {}
        x = self.data_x[index]
        y = self.data_y[index]


        if self.task != 'task-3':
            x = x.replace('[CLS]', '').replace('[SEP]', '</s>')
            x = self.tokenizer(x, return_tensors='pt', max_length=self.padding_len, truncation=True, padding='max_length')
            x['input_ids'] = x['input_ids'].squeeze()
            x['attention_mask'] = x['attention_mask'].squeeze()
            if len(x) == 3:
                x['token_type_ids'] = x['token_type_ids'].squeeze()
        else:
            # transform to a list
            if isinstance(index, int):
                x, y  = [x], [y]
            x, y = utils.vit5_encode(x, y, self.padding_len, self.target_len, self.tokenizer)

        item.update({'input': x})
        item.update({'label': y})

        return item