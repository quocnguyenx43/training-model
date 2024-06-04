from torch.utils.data import Dataset
import torch
import torch.nn.functional as F


class RecruitmentDataset(Dataset):

    def __init__(self, df, tokenizer, padding_length, type):
        self.df = df

        self.tokenizer = tokenizer 
        self.padding_length = padding_length
        self.type = type

        self.X = self.create_X()
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = {}
        x = self.X[index]
        x = x.replace('[CLS]', '').replace('[SEP]', '</s>')
        y = self.df.loc[index]

        x = self.tokenizer(x, return_tensors='pt', max_length=self.padding_length, truncation=True, padding='max_length')
        x['input_ids'] = x['input_ids'].squeeze()
        x['attention_mask'] = x['attention_mask'].squeeze()
        if len(x) == 3:
            x['token_type_ids'] = x['token_type_ids'].squeeze()

        if self.type == 'task_1':
            label = F.one_hot(torch.tensor(int(y['label'])), 3).float()
        elif self.type == 'task_2':
            label = y[['title_aspect', 'desc_aspect', 'company_aspect', 'other_aspect']].astype(int)
            label = torch.tensor(label)
            label = torch.eye(4)[label]
            label = label.squeeze(dim=1).float()

        item.update({'input': x})
        item.update({'label': label})

        return item

    def create_X(self):
        infor_cols = [
            'tiêu đề', 'ngành nghề',
            'mô tả', 'kinh nghiệm', 'học vấn', 'bằng cấp', 'quyền lợi',
            'tên công ty', 'địa chỉ', 'số điện thoại', 'người liên hệ',
            # 'tên người đăng', 'số điện thoại người đăng', 'ngày đăng', 'hạn nộp CV', 'tin đăng ẩn danh', 'là nhà tuyển dụng',
            'số lượng tuyển', 'hình thức hợp đồng', 'hình thức trả lương', 'lương tối thiểu', 'lương tối đa', 'giới tính', 'năm sinh', 'tuổi', 'tuổi thấp nhất', 'tuổi cao nhất',
        ]

        X_data = self.df[[
            'title', 'job_type', 'body', 'experience', 'education', 'certification',
            'benefit', 'company_name', 'location', 'phone', 'contact_name',
            'vacancy', 'contact_type', 'salary_type', 'min_salary', 'max_salary',
            'gender', 'year_of_birth', 'age', 'min_age', 'max_age'
        ]]

        X_df_combined = []
        for _, row in X_data.iterrows():
            inp = '[CLS] '
            for i, j in enumerate(row):
                inp = inp + infor_cols[i] + ": " + str(j) + " [SEP] "

            X_df_combined.append(inp)

        return X_df_combined