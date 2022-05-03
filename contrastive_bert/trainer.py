'''
    File name: trainer.py
    Author: Gabriel Moreira
    Date last modified: 03/08/2022
    Python Version: 3.7.10
'''

import os
import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from utils import Tracker, f1_score

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

class Trainer:
    def __init__(
        self,
        model,
        epochs,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        dev_loader,
        device,
        name,
        resume):

        self.model        = model
        self.epochs       = epochs
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.criterion    = criterion
        self.train_loader = train_loader
        self.dev_loader   = dev_loader
        self.device       = device
        self.name         = name
        self.start_epoch  = 1

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler()

        self.tracker = Tracker(['epoch',
                                'train_loss',
                                'train_f1',
                                'dev_loss',
                                'dev_f1',
                                'lr'], name, load=resume)

        if resume:
            self.resume_checkpoint()

    def fit(self):
        is_best = False
        for epoch in range(self.start_epoch, self.epochs+1):
            train_loss, train_f1 = self.train_epoch(epoch)
            dev_loss,   dev_f1   = self.validate_epoch()

            self.epoch_verbose(epoch, train_loss, train_f1, dev_loss, dev_f1)
            # Check if better than previous models
            if epoch > 1:
                is_best = self.tracker.isSmaller('dev_loss', dev_loss)
            else:
                is_best = True

            self.tracker.update(epoch=epoch,
                                train_loss=train_loss,
                                train_f1=train_f1,
                                dev_loss=dev_loss,
                                dev_f1=dev_f1,
                                lr=self.optimizer.param_groups[0]['lr'])
            self.save_checkpoint(epoch, is_best)


    def train_epoch(self, epoch):
        # Set model to training mode
        self.model.train()

        # Progress bar over the current epoch
        batch_bar = tqdm(total=len(self.train_loader), dynamic_ncols=True, desc='Train') 

        # Cumulative loss over all batches or Avg Loss * num_batches
        total_loss_epoch = 0
        total_f1 = 0.0

        accumulation_steps = 64

        # Iterate one batch at a time
        self.optimizer.zero_grad()
        for i_batch, (tokens, idx) in enumerate(self.train_loader):
            # tokens[key] for all keys has shape torch.Size(num_sentences, max_sentence_length)
            with torch.cuda.amp.autocast():
                # output['pooler_output'] has shape torch.Size (num_sentences, 768)
                output = self.model(**tokens)
                # embeddings has shape torch.Size (num_sentences, 768)
                embeddings = F.normalize(output['pooler_output'], p=2, dim=1)
                # energy has shape torch.Size([num_sources,1])
                energy = torch.matmul(embeddings[1:,:], torch.transpose(embeddings[0:1,:],1,0)).flatten()
                energy = F.softmax(energy, dim=0)
                positive_idx = np.arange(0, idx['neg_start_idx']-1)
                loss_batch = self.criterion(energy, positive_idx) / accumulation_steps

            self.scaler.scale(loss_batch).backward()
            if i_batch % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.optimizer.zero_grad()
                self.scaler.update()
                self.scheduler.step()

            # Performance metrics
            total_loss_epoch += loss_batch.detach()
            avg_loss_epoch    = float((total_loss_epoch*accumulation_steps) / (i_batch + 1))

            top_values, top_idx    = torch.topk(energy.detach(), 5, sorted=True)
            top_values             = top_values.cpu().tolist()
            top_idx                = top_idx.cpu().tolist()
            retrieved_idx          = top_idx[0:1]
            retrieved              = len(retrieved_idx)
            retrieved_and_relevant = len(set(retrieved_idx).intersection(set(positive_idx)))
            relevant               = len(positive_idx)

            total_f1     += f1_score(retrieved_and_relevant, retrieved, relevant)
            avg_f1_epoch  = float(total_f1 / (i_batch + 1))

            # Performance tracking verbose
            batch_bar.set_postfix(
                avgloss="{:0.5f}".format(avg_loss_epoch),
                energy="{}".format(np.array(top_values)),
                idx="{}".format(np.array(top_idx)),
                avgf1="{:.1f}".format(avg_f1_epoch),
                lr="{:1.2e}".format(float(self.optimizer.param_groups[0]['lr'])))
            batch_bar.update()

        batch_bar.close()

        return avg_loss_epoch, avg_f1_epoch


    def validate_epoch(self):
        # Set model to evaluation mode
        self.model.eval()

        total_f1   = 0.0
        total_loss = 0.0

        batch_bar = tqdm(total=len(self.dev_loader), dynamic_ncols=True, desc='Dev') 
        # Do not store gradients 
        with torch.no_grad():
            # Get batches from DEV loader
            for i_batch, (tokens, idx) in enumerate(self.dev_loader):
                # output['pooler_output'] has shape torch.Size (num_sentences, 768)
                output = self.model(**tokens)
                # embeddings has shape torch.Size (num_sentences, 768)
                embeddings = F.normalize(output['pooler_output'], p=2, dim=1)
                # energy has shape torch.Size([num_sources,1])
                energy = torch.matmul(embeddings[1:,:], torch.transpose(embeddings[0:1,:],1,0)).flatten()
                energy = F.softmax(energy, dim=0)
                positive_idx = np.arange(0, idx['neg_start_idx']-1)
                loss_batch   = self.criterion(energy, positive_idx)
                total_loss  += loss_batch.detach()

                top_values, top_idx    = torch.topk(energy.detach(), 5, sorted=True)
                top_values             = top_values.cpu().tolist()
                top_idx                = top_idx.cpu().tolist()
                retrieved_idx          = top_idx[0:1]
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

        return avg_loss, f1


    def save_checkpoint(self, epoch, is_best):
        '''
            Save model dict and hyperparams
        '''
        # Save best model weights
        if is_best:
            best_path = os.path.join(self.name, "best_weights.pth")
            torch.save(self.model.state_dict(), best_path)
            print("Saving best model: {}".format(best_path))


    def resume_checkpoint(self):
        '''
        '''
        resume_path = os.path.join(self.name, "checkpoint.pth")
        print("Loading checkpoint: {} ...".format(resume_path))

        checkpoint       = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.model       = checkpoint["model"]
        self.optimizer   = checkpoint["optimizer"]

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    
    def epoch_verbose(self, epoch, train_loss, train_f1, dev_loss, dev_f1):
        log = "\nEpoch: {}/{} summary:".format(epoch, self.epochs)
        log += "\n            Avg train Xent  |  {:.6f}".format(train_loss)
        log += "\n            Avg train F1    |  {:.6f}".format(train_f1)
        log += "\n            Avg dev Xent    |  {:.6f}".format(dev_loss)
        log += "\n            Avg dev F1      |  {:.6f}".format(dev_f1)
        print(log)