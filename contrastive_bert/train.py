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
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from loader import QASamples
from loss import ContrastiveLoss
from trainer import Trainer
from utils import getNumTrainableParams

if __name__ == '__main__':
    cfg = {'name'       : 'v4',    # <---- name of your model here
           'seed'       : 1,
           'epochs'     : 4,
           'batch_size' : 1,
           'lr'         : 1e-5,
           'resume'     : False}

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # If experiment folder doesn't exist create it
    if not os.path.isdir(cfg['name']):
        os.makedirs(cfg['name'])
        print("Created experiment folder : ", cfg['name'])
    else:
        print(cfg['name'], "folder already exists.")

    DATA_PATH  = "./data/WebQA_train_val.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Running on device: {}".format(device))

    configuration = AutoConfig.from_pretrained('bert-base-uncased')
    configuration.hidden_dropout_prob = 0.3
    configuration.attention_probs_dropout_prob = 0.3

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model     = BertModel.from_pretrained("bert-base-uncased", config=configuration).cuda()
    model.load_state_dict(torch.load('./v3/best_weights.pth'))

    train_samples = QASamples(DATA_PATH, 'train', tokenizer)
    train_loader  = DataLoader(train_samples,
                               batch_size=1,
                               shuffle=True,
                               collate_fn=train_samples.collate_fn)

    dev_samples = QASamples(DATA_PATH, 'val', tokenizer)
    dev_loader  = DataLoader(dev_samples,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=dev_samples.collate_fn)
 
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(train_loader)*cfg['epochs']/16)
    trainer = Trainer(model,
                      cfg['epochs'],
                      optimizer,
                      scheduler,
                      criterion,
                      train_loader, 
                      dev_loader,
                      device,
                      cfg['name'],
                      cfg['resume'])

    # Verbose
    print('Experiment ' + cfg['name'])
    print('Running on', device)
    print('Train - {} batches of size {}'.format(len(train_loader), cfg['batch_size']))
    print('  Val - {} batches of size {}'.format(len(dev_loader), cfg['batch_size']))
    print('Number of trainable parameters: {}'.format(getNumTrainableParams(model)))
    print(model)

    trainer.fit() 
