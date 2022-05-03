'''
    File name: train.py
    Author: Gabriel Moreira
    Date last modified: 03/08/2022
    Python Version: 3.7.10
'''

import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

from loader import QASamples
from loss import ContrastiveLoss
from trainer import Trainer
from utils import getNumTrainableParams
from tqdm import tqdm
from utils import f1_score


if __name__ == '__main__':
    DATA_PATH  = "./data/WebQA_train_val.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Running on device: {}".format(device))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model     = BertModel.from_pretrained("bert-base-uncased").cuda()
    model.load_state_dict(torch.load('./v3/best_weights.pth'))

    dev_samples = QASamples(DATA_PATH, 'val', tokenizer)
    dev_loader  = DataLoader(dev_samples,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=dev_samples.collate_fn)
 
    criterion = ContrastiveLoss()

    model.eval()

    total_f1   = 0.0
    total_loss = 0.0

    threshold = 0.06
    batch_bar = tqdm(total=len(dev_loader), dynamic_ncols=True, desc='Dev') 
    # Do not store gradients 
    with torch.no_grad():
        # Get batches from DEV loader
        for i_batch, (tokens, idx) in enumerate(dev_loader):
            # output['pooler_output'] has shape torch.Size (num_sentences, 768)
            output = model(**tokens)
            # embeddings has shape torch.Size (num_sentences, 768)
            embeddings = F.normalize(output['pooler_output'], p=2, dim=1)
            # energy has shape torch.Size([num_sources,1])
            energy = torch.matmul(embeddings[1:,:], torch.transpose(embeddings[0:1,:],1,0)).flatten()
            energy = F.softmax(energy, dim=0)
            positive_idx = np.arange(0, idx['neg_start_idx']-1)
            loss_batch   = criterion(energy, positive_idx)
            total_loss  += loss_batch.detach()

            energy = energy.flatten().detach().cpu().numpy()
            retrieved_idx = list(np.where(energy > threshold)[0])
            retrieved              = len(retrieved_idx)
            retrieved_and_relevant = len(set(retrieved_idx).intersection(set(positive_idx)))
            relevant               = len(positive_idx)
            total_f1               += f1_score(retrieved_and_relevant, retrieved, relevant)

            batch_bar.set_postfix(
                    avgloss="{:0.5f}".format(total_loss / (i_batch + 1)),
                    avgf1="{:.1f}".format(total_f1 / (i_batch + 1)))
            batch_bar.update()

    batch_bar.close()
    avg_loss = float(total_loss / (i_batch + 1))
    f1       = float(total_f1 / (i_batch + 1))

    print("Dev loss = {}".format(avg_loss))
    print("F1       = {}".format(f1))