# coding: utf-8
import os
import time
import numpy as np
import pandas as pd
from sklearn import metrics
import datetime
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import Model


class Solver(object):
    def __init__(self, data_loader, config):
        # data loader
        self.input_length = config.input_length
        self.data_loader = data_loader
        self.dataset = config.dataset
        self.data_path = config.data_path

        # model hyper-parameters
        self.conv_channels = config.conv_channels
        self.sample_rate = config.sample_rate
        self.n_fft = config.n_fft
        self.n_harmonic = config.n_harmonic
        self.semitone_scale = config.semitone_scale
        self.learn_bw = config.learn_bw

        # training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.use_tensorboard = config.use_tensorboard

        # model path and step size
        self.model_save_path = config.model_save_path
        self.model_load_path = config.model_load_path
        self.log_step = config.log_step
        self.batch_size = config.batch_size

        # cuda
        self.is_cuda = torch.cuda.is_available()

        # Build model
        self.get_dataset()
        self.build_model()

    def get_dataset(self):
        if self.dataset == 'mtat':
            self.valid_list = np.load(os.path.join(self.data_path, 'mtat', 'valid.npy'))
            self.binary = np.load(os.path.join(self.data_path, 'mtat', 'binary.npy'))
        elif self.dataset == 'dcase':
            df = pd.read_csv(os.path.join(self.data_path, 'dcase', 'df.csv'), delimiter='\t', names=['file', 'start', 'end', 'path', 'split', 'label'])
            df = df[df['split'] == 'val']
            self.valid_list = list(df['path'])
            self.binary = list(df['label'])
        elif self.dataset == 'keyword':
            from data_loader.keyword_loader import get_audio_loader
            self.valid_loader = get_audio_loader(self.data_path, self.batch_size, input_length = self.input_length, tr_val='val')

    def get_model(self):
        return Model(conv_channels=self.conv_channels,
                     sample_rate=self.sample_rate,
                     n_fft=self.n_fft,
                     n_harmonic=self.n_harmonic,
                     semitone_scale=self.semitone_scale,
                     learn_bw=self.learn_bw,
                     dataset=self.dataset)

    def build_model(self):
        # model
        self.model = self.get_model()

        # cuda
        if self.is_cuda:
            self.model.cuda()

        # load pretrained model
        if len(self.model_load_path) > 1:
            self.load(self.model_load_path)

        # optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=1e-4)

    def load(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def get_loss_function(self):
        if self.dataset == 'mtat' or self.dataset == 'dcase':
            return nn.BCELoss()
        elif self.dataset == 'keyword':
            return nn.CrossEntropyLoss()

    def train(self):
        # Start training
        start_t = time.time()
        current_optimizer = 'adam'
        reconst_loss = self.get_loss_function()
        best_metric = 0
        drop_counter = 0
        for epoch in range(self.n_epochs):
            # train
            ctr = 0
            drop_counter += 1
            self.model.cuda()
            self.model.train()
            for x, y in self.data_loader:
                ctr += 1
                # Forward
                x = self.to_var(x)
                y = self.to_var(y)
                out = self.model(x)

                # Backward
                loss = reconst_loss(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log
                if (ctr) % self.log_step == 0:
                    print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                epoch+1, self.n_epochs, ctr, len(self.data_loader), loss.item(),
                                datetime.timedelta(seconds=time.time()-start_t)))

            # validation
            if self.dataset == 'mtat':
                roc_auc, pr_auc, loss = self.get_validation_score()
                score = 1 - loss
                if score > best_metric:
                    print('best model!')
                    best_metric = score
                    torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'best_model.pth'))
            elif self.dataset == 'dcase':
                if epoch > 10:
                    f1, loss = self.get_validation_score()
                    score = 1 - loss
                    score = f1
                else:
                    score = 0
                if score > best_metric:
                    print('best model!')
                    best_metric = score
                    torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'best_model.pth'))
            elif self.dataset == 'keyword':
                acc, loss = self.get_validation_acc()
                score = 1 - loss
                if score > best_metric:
                    print('best model: %.4f' % acc)
                    best_metric = score
                    torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'best_model.pth'))

            # schedule optimizer
            current_optimizer, drop_counter = self.opt_schedule(current_optimizer, drop_counter)

        print("[%s] Train finished. Elapsed: %s"
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.timedelta(seconds=time.time() - start_t)))

    def opt_schedule(self, current_optimizer, drop_counter):
        # adam to sgd
        if current_optimizer == 'adam' and drop_counter == 60:
            self.load = os.path.join(self.model_save_path, 'best_model.pth')
            self.optimizer = torch.optim.SGD(self.model.parameters(), 0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
            current_optimizer = 'sgd_1'
            drop_counter = 0
            print('sgd 1e-3')
        # first drop
        if current_optimizer == 'sgd_1' and drop_counter == 20:
            self.load = os.path.join(self.model_save_path, 'best_model.pth')
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.0001
            current_optimizer = 'sgd_2'
            drop_counter = 0
            print('sgd 1e-4')
        # second drop
        if current_optimizer == 'sgd_2' and drop_counter == 20:
            self.load = os.path.join(self.model_save_path, 'best_model.pth')
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.00001
            current_optimizer = 'sgd_3'
            print('sgd 1e-5')
        return current_optimizer, drop_counter

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'model': model}, filename)

    def get_auc(self, est_array, gt_array):
        roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
        print('roc_auc: %.4f' % roc_aucs)
        print('pr_auc: %.4f' % pr_aucs)
        return roc_aucs, pr_aucs

    def get_tensor(self, fn):
        # load audio
        if self.dataset == 'mtat':
            npy_path = os.path.join(self.data_path, 'mtat', 'npy', fn.split('/')[1][:-3]) + 'npy'
        elif self.dataset == 'dcase':
            npy_path = fn
        raw = np.load(npy_path, mmap_mode='r')

        # split chunk
        length = len(raw)
        if length < self.input_length:
            nnpy = np.zeros(self.input_length)
            ri = int(np.floor(np.random.random(1) * (self.input_length - length)))
            nnpy[ri:ri+length] = raw
            raw = nnpy
            length = len(raw)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(raw[i*hop:i*hop+self.input_length]).unsqueeze(0)
        return x

    def get_validation_score(self):
        self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = self.get_loss_function()
        index = 0
        for line in tqdm.tqdm(self.valid_list):
            if self.dataset == 'mtat':
                ix, fn = line.split('\t')
            elif self.dataset == 'dcase':
                fn = line

            # load and split
            x = self.get_tensor(fn)

            # ground truth
            if self.dataset == 'mtat':
                ground_truth = self.binary[int(ix)]
            elif self.dataset == 'dcase':
                ground_truth = np.fromstring(self.binary[index][1:-1], dtype=np.float32, sep=' ')

            # forward
            x = self.to_var(x)
            y = torch.tensor([ground_truth.astype('float32') for i in range(self.batch_size)]).cuda()
            out = self.model(x)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()

            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)

            gt_array.append(ground_truth)
            index += 1

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)
        print('loss: %.4f' % loss)

        if self.dataset == 'mtat':
            roc_auc, pr_auc = self.get_auc(est_array, gt_array)
            return roc_auc, pr_auc, loss
        elif self.dataset == 'dcase':
            prd_array = (est_array > 0.1).astype(np.float32)
            f1 = metrics.f1_score(gt_array, prd_array, average='samples')
            print('f1: %.4f' % f1)
            return f1, loss

    def get_validation_acc(self):
        self.model.eval()
        reconst_loss = self.get_loss_function()
        est_array = []
        gt_array = []
        losses = []
        for x, y in tqdm.tqdm(self.valid_loader):
            x = self.to_var(x)
            y = self.to_var(y)
            out = self.model(x)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()
            y = y.detach().cpu()
            _prd = [int(np.argmax(prob)) for prob in out]
            for i in range(len(_prd)):
                est_array.append(_prd[i])
                gt_array.append(y[i])
        est_array = np.array(est_array)
        gt_array = np.array(gt_array)
        acc = metrics.accuracy_score(gt_array, est_array)
        loss = np.mean(losses)
        print('accuracy: %.4f' % acc)
        print('loss: %.4f' % loss)
        return acc, loss
