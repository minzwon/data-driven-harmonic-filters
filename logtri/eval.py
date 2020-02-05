# coding: utf-8
import os
import time
import numpy as np
import datetime
import tqdm
import fire
import argparse
from sklearn import metrics

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import Model


class Predict(object):
    def __init__(self, config):
        # data loader
        self.dataset = config.dataset
        self.data_path = config.data_path

        # model hyper-parameters
        self.conv_channels = config.conv_channels
        self.sample_rate = config.sample_rate
        self.semitone_scale = config.semitone_scale
        self.n_fft = config.n_fft
        self.n_harmonic = config.n_harmonic
        self.learn_f0 = config.learn_f0
        self.learn_bw = config.learn_bw
        self.input_length = config.input_length

        # training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.use_tensorboard = config.use_tensorboard

        # model path and step size
        self.model_load_path = config.model_load_path
        self.batch_size = config.batch_size

        # cuda
        self.is_cuda = torch.cuda.is_available()

        # Build model
        self.build_model()

        # Start with trained model
        self.load(config.model_load_path)

    def get_model(self):
        return Model(conv_channels=self.conv_channels,
                      sample_rate=self.sample_rate,
                      n_fft=self.n_fft,
                      n_harmonic=self.n_harmonic,
                      semitone_scale=self.semitone_scale,
                      learn_f0=self.learn_f0,
                      learn_bw=self.learn_bw,
                      dataset=self.dataset)

    def build_model(self):
        # model and optimizer
        self.model = self.get_model()

        # cuda
        if self.is_cuda == True:
            self.model.cuda()

    def load(self, filename):
        S = torch.load(filename)
        #temp = S['hconv.spec.mel_scale.fb']
        #del(S['hconv.spec.mel_scale.fb'])
        self.model.load_state_dict(S)
        #S['hconv.spec.mel_scale.fb'] = temp
        print(self.model.hconv.bw_Q)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def detach(self, x):
        return Variable(x.data)

    def get_tensor(self, fn, num_chunks):
        if self.dataset == 'mtat':
            # load audio
            raw_path = os.path.join(self.data_path, 'raw', fn.split('/')[1][:-3]) + 'npy'
            raw = np.load(raw_path, mmap_mode='r')
            # split chunk
            length = len(raw)
            chunk_length = self.input_length
            hop = (length - chunk_length) // num_chunks
            x = torch.zeros(num_chunks, chunk_length)
            for i in range(num_chunks):
                x[i] = torch.Tensor(raw[i*hop:i*hop+chunk_length]).unsqueeze(0)
            return x

    def forward(self, x):
        x = self.to_var(x)
        x = self.model(x)
        x = self.detach(x)
        return x.cpu()

    def get_auc(self, est_array, gt_array):
        est_array = np.array(est_array)
        gt_array = np.array(gt_array)

        roc_auc = []
        pr_auc = []
        for _tag in range(50):
            roc_auc.append(metrics.roc_auc_score(gt_array[:, _tag], est_array[:, _tag]))
            pr_auc.append(metrics.average_precision_score(gt_array[:, _tag], est_array[:, _tag]))
        print('roc_auc: %.4f' % np.mean(roc_auc))
        print('pr_auc: %.4f' % np.mean(pr_auc))

    def evaluate(self, num_chunks=16):
        self.evaluate_auto_tagging(num_chunks)

    def evaluate_auto_tagging(self, num_chunks=16):
        self.model.eval()
        if self.dataset == 'mtat':
            filelist = np.load(os.path.join(self.data_path, 'data/test_new.npy'))
        binary = np.load(os.path.join(self.data_path, 'data/binary.npy'))

        est_array = []
        gt_array = []
        for line in tqdm.tqdm(filelist):
            ix, fn = line.split('\t')
            # load and split
            x = self.get_tensor(fn, num_chunks)

            # forward
            prd = self.forward(x)

            # estimated
            estimated = np.array(prd).mean(axis=0)
            est_array.append(estimated)

            # ground truth
            ground_truth = binary[int(ix)]
            gt_array.append(ground_truth)

        # get roc_auc and pr_auc
        self.get_auc(est_array, gt_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model hyper-parameters
    parser.add_argument('--conv_channels', type=int, default=128)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--n_fft', type=int, default=512)
    parser.add_argument('--n_harmonic', type=int, default=6)
    parser.add_argument('--semitone_scale', type=int, default=2)
    parser.add_argument('--bw_scale', type=float, default=4.0)
    parser.add_argument('--learn_f0', type=bool, default=False)
    parser.add_argument('--learn_bw', type=str, default=False)

    # dataset
    parser.add_argument('--input_length', type=int, default=49152)
    parser.add_argument('--dataset', type=str, default='mtat', choices=['mtat', 'dcase', 'keyword'])

    # training settings
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=int, default=1)
    parser.add_argument('--model_save_path', type=str, default='./../models')
    parser.add_argument('--model_load_path', type=str, default='./../models/best_model.pth')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--log_step', type=int, default=20)

    config = parser.parse_args()
    print(config)

    p = Predict(config)
    p.evaluate()






