# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modules import HarmonicConv, MelSpec, ResNet_mp, ResNet_3D

class Model(nn.Module):
    def __init__(self,
                conv_channels=128,
                sample_rate=16000,
                n_fft=512,
                n_harmonic=6,
                semitone_scale=2,
                learn_f0=False,
                learn_bw=None,
                dataset='mtat'):
        super(Model, self).__init__()

        # harmonic convolution
#        self.hconv = HarmonicConv(sample_rate=sample_rate,
#                                 n_fft=n_fft,
#                                 n_harmonic=n_harmonic,
#                                 semitone_scale=semitone_scale,
#                                 learn_f0=learn_f0,
#                                 learn_bw=learn_bw)
        self.hconv = MelSpec()
        self.hconv_bn = nn.BatchNorm2d(1)

        # 2D CNN
        self.conv_2d = ResNet_mp(input_channels=1,
                                   conv_channels=conv_channels)

    def forward(self, x):
        # harmonic convolution
        x = self.hconv(x)
        x = self.hconv_bn(x.unsqueeze(1))

        # 2D CNN
        logits = self.conv_2d(x)

        return logits
