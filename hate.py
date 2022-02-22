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

import logging
from transformers import get_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs

param = DistributedDataParallelKwargs(find_unused_parameters=False, check_reduction=False)
accelerator = Accelerator(fp16=False, kwargs_handlers=[param])
device = accelerator.device

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

# checkpoint = "klue/bert-base"
# model = AutoModel.from_pretrained(checkpoint)
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 7e-4
num_epochs = 100
checkpoint = './bert-base'
model = pr_model()
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
    def __init__(self, train_path, valid_path, tokenizer):
        super().__init__()
        tr = pd.read_csv(train_path)
        val = pd.read_csv(valid_path)
        self.data = pd.concat([tr, val], ignore_index=True)
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sent = self.data.iloc[index, 0]
        tokenized_text = self.tokenizer(sent, return_tensors='pt')
        target = self.data.iloc[index, 1]
        if target == 'none':
            target = torch.LongTensor([1])
        elif target == 'offensive':
            target = torch.LongTensor([2])
        else:
            target = torch.LongTensor([3])
        length = (v.shape[1] for k, v in tokenized_text.items())
        return tokenized_text, length, target

def SNS_collate(batch):
    batch, length, target = zip(*batch)
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
    label_batch = torch.cat(target, dim=0)
    # print({'input_ids':ids_batch, 'token_type_ids':label_batch, 'attention_mask':mask_batch})
    return {'input_ids':ids_batch, 'token_type_ids':type_id_batch, 'attention_mask':mask_batch}, label_batch


tr_dataset = SNS_Dataset(train_path, valid_path, tokenizer)
# val_dataset = SNS_Dataset(valid_path, tokenizer)
# for line in dataset:
#     print(line)
#     break

dataloader = DataLoader(tr_dataset,
                        batch_size=128,
                        shuffle=True,
                        num_workers=8,
                        collate_fn=SNS_collate)

# valid_dataloader = DataLoader(val_dataset,
#                         batch_size=128,
#                         shuffle=False,
#                         num_workers=8,
#                         collate_fn=SNS_collate)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
num_training_steps = num_epochs * len(dataloader)

def train(model, dataloader, optimizer, criterion, lr_scheduler, accelerator):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for sent, tgt in dataloader:
        # sent, tgt = {k:v.to(device) for k, v in sent.items()}, tgt.to(device)
        optimizer.zero_grad()
        # print({v.shape for k, v in sent.items()})
        output = model(sent)
        # print(output.shape, tgt.shape)
        loss = criterion(output, tgt)
        accelerator.backward(loss)
        # loss.backward()
        optimizer.step()
        lr_scheduler.step()
        epoch_loss += loss.item()

        pred = torch.max(output, 1)[1].type(torch.float) # (batch, max idx)
        acc = torch.sum((pred == tgt) / pred.shape[0], dim=0).item()
        epoch_acc += acc
    N = len(dataloader)
    return epoch_loss / N, epoch_acc/ N


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for sent, tgt in dataloader:
            # sent, tgt = {k:v.to(device) for k, v in sent.items()}, tgt.to(device)
            output = model(sent)
            loss = criterion(output, tgt)
            epoch_loss += loss.item()

            pred = torch.max(output, 1)[1].type(torch.float) # (batch, max idx)
            acc = torch.sum((pred == tgt) / pred.shape[0], dim=0).item()
            epoch_acc += acc
        N = len(dataloader)
    return epoch_loss / N, epoch_acc/ N


def get_logger(name: str, file_path: str, stream=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 시간, 로거 이름, 로깅 레벨, 메세지
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    # console에 출력하도록 설정
    stream_handler = logging.StreamHandler()
    # 현재 디렉토리에 파일로 로깅하도록 설정
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    # 현재 디렉토리에 로깅 저장
    logger.addHandler(file_handler)

    return logger

def save(filename, model, logger):
    save_model = accelerator.unwrap_model(model)
    state = {
        'model': save_model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    accelerator.save(state, filename)
    logger.info('Model saved')



lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=8000,
        num_training_steps=num_training_steps
    )

logger = get_logger(name='train',
                    file_path=os.path.join('.', 'train_log.log'),
                    stream=True)

dataloader, model, optimizer = accelerator.prepare(
    dataloader,  model, optimizer
    )

for epoch in range(num_epochs):
    tr_loss, tr_acc = train(model, dataloader, optimizer, criterion, lr_scheduler, accelerator)
    logger.info('Epoch %d (Training) Loss %0.8f, Acc %0.8f' % (epoch, tr_loss, tr_acc))
    
    # val_loss, val_acc = evaluate(model, valid_dataloader, criterion)
    # logger.info('Epoch %d (Evaluate) Loss %0.8f Acc %0.8f' % (epoch, val_loss, val_acc))
    save(os.path.join('checkpoint', f"model_{epoch:03d}.pt"), model, logger)
                                                           
