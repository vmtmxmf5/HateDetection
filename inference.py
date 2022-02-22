from transformers import AutoModel, AutoTokenizer, BertTokenizer
from torch.utils.data import Dataset, DataLoader

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import pandas as pd


class pr_model(nn.Module):
    def __init__(self, label_class:int = 4, hidden_size:int = 768, checkpoint:str = './bert-base'):
        super().__init__()
        self.linear = nn.Linear(hidden_size, label_class)
        self.bert = AutoModel.from_pretrained(checkpoint, local_files_only=True)
    def forward(self, inputs):
        logits = self.bert(**inputs)  # logits : (batch, seq_len, hidden size)
        cls_token = logits[1] # cls_token : (batch, hidden size)
        class_dist = F.softmax(self.linear(cls_token), dim=-1)
        return class_dist


def load(filename, model):
    # state = torch.load(filename)
    state = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])

# checkpoint = "klue/bert-base"
# model = AutoModel.from_pretrained(checkpoint)
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 7e-4
num_epochs = 30
checkpoint = './bert-base'
model = pr_model()
model.to(device)
load('./checkpoint/model_070.pt', model)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)

# tmp = tokenizer('[CLS] 안녕하시렵니까리오 [SEP]')
# tmp2 = tokenizer('[CLS]안녕하시렵니까리오[SEP]')
# print(tmp)
# print(tmp2)
# tmp = tokenizer.tokenize('[CLS] 안녕하시렵니까리오 [SEP]')
# print(tmp)
# print(tokenizer.convert_tokens_to_ids(tmp))

train_path = 'train.hate.csv'
valid_path = 'dev.hate.csv'


class SNS_Dataset(Dataset):
    def __init__(self, path, tokenizer):
        super().__init__()
        self.data = pd.read_csv(path)
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sent = self.data.iloc[index, 0]
        tokenized_text = self.tokenizer(sent, return_tensors='pt')
        length = (v.shape[1] for k, v in tokenized_text.items())
        return tokenized_text, length

def SNS_collate(batch):
    batch, length = zip(*batch)
    ids, mask, type_id = zip(*length)
    max_ids, max_mask, max_type_id = max(ids), max(mask), max(type_id)
    ids, mask, type_id = list(ids), list(mask), list(type_id)

    ids_res, mask_res, type_id_res = [], [], []
    for i, sample in enumerate(batch):
        len_ids = max_ids - ids[i]
        len_mask = max_mask - mask[i]
        len_type_id = max_type_id - type_id[i]
        ids_tensor = torch.cat([sample['input_ids'], torch.LongTensor([[tokenizer.pad_token_id] * len_ids])], dim=1)
        mask_tensor = torch.cat([sample['attention_mask'], torch.LongTensor([[0] * len_mask])], dim=1)
        type_tensor = torch.cat([sample['token_type_ids'], torch.LongTensor([[tokenizer.pad_token_id] * len_type_id])], dim=1)
        ids_res.append(ids_tensor)
        mask_res.append(mask_tensor)
        type_id_res.append(type_tensor)
    ids_batch = torch.cat(ids_res, dim=0)
    mask_batch = torch.cat(mask_res, dim=0)
    type_id_batch = torch.cat(type_id_res, dim=0)
    # print({'input_ids':ids_batch, 'token_type_ids':label_batch, 'attention_mask':mask_batch})
    return {'input_ids':ids_batch, 'token_type_ids':type_id_batch, 'attention_mask':mask_batch}


dataset = SNS_Dataset('test.hate.no_label.csv', tokenizer)
# for line in dataset:
#     print(line)
#     break

dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=SNS_collate)


def evaluate(model, dataloader):
    model.eval()
    sentences = pd.read_csv('test.hate.no_label.csv')
    df = pd.DataFrame(columns=['comments', 'label'])
    with torch.no_grad():
        for i, sent in enumerate(dataloader):
            sent = {k:v.to(device) for k, v in sent.items()}
            output = model(sent)
            # loss = criterion(output, tgt)
            # epoch_loss += loss.item()

            pred = torch.max(output, 1)[1].type(torch.float) # (batch, max idx)
            # acc = torch.sum((pred == tgt) / pred.shape[0], dim=0).item()
            # epoch_acc += acc
            # print(pred.item())
            tmp = int(pred.item())
            if tmp == 1:
                df.loc[i] = [sentences.iloc[i, 0], 0]
            elif tmp == 2:
                df.loc[i] = [sentences.iloc[i, 0], 1]
            else:
                df.loc[i] = [sentences.iloc[i, 0], 2]                
    return df.sort_index()

results = evaluate(model, dataloader)
print(results)
results.to_csv('./results.csv', index=False, encoding='utf-8')
