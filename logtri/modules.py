import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import sys
from torch.autograd import Variable
import math
import librosa

def hz_to_midi(hz):
    return 12 * (torch.log2(hz) - np.log2(440.0)) + 69

def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0)/12.0))

def note_to_hz(note):
    return librosa.core.note_to_hz(note)

def note_to_midi(note):
    return librosa.core.note_to_midi(note)

def hz_to_note(hz):
    return librosa.core.hz_to_note(hz)

def initialize_filterbank(sample_rate, n_harmonic, semitone_scale):
    # MIDI
    # lowest note
    low_midi = note_to_midi('C1')

    # highest note
    high_note = hz_to_note(sample_rate / (2 * n_harmonic))
    high_midi = note_to_midi(high_note)

    # number of scales
    level = (high_midi - low_midi) * semitone_scale
    midi = np.linspace(low_midi, high_midi, level + 1)
    hz = midi_to_hz(midi[:-1])

    # stack harmonics
    harmonic_hz = []
    for i in range(n_harmonic):
        harmonic_hz = np.concatenate((harmonic_hz, hz * (i+1)))

    return harmonic_hz, level


class HarmonicConv(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=513,
                 win_length=None,
                 hop_length=None,
                 pad=0,
                 power=2,
                 normalized=False,
                 n_harmonic=6,
                 semitone_scale=2,
                 bw_Q=3.0,
                 learn_f0=False,
                 learn_bw=None):
        super(HarmonicConv, self).__init__()

        # Parameters
        self.sample_rate = sample_rate
        self.n_harmonic = n_harmonic

        # Spectrogram
        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length,
                                                      hop_length=None, pad=0,
                                                      window_fn=torch.hann_window,
                                                      power=power, normalized=normalized, wkwargs=None)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Initialize the filterbank. Equally spaced in MIDI scale.
        harmonic_hz, self.level = initialize_filterbank(sample_rate, n_harmonic, semitone_scale)

        # Center frequncies to tensor
        self.f0 = torch.tensor(harmonic_hz.astype('float32'))
        if learn_bw == 'only_Q':
            self.bw_Q = nn.Parameter(torch.tensor(np.array([bw_Q]).astype('float32')))
        else:
            self.bw_Q = torch.tensor(np.array([bw_Q]).astype('float32'))

    def get_harmonic_fb(self):
        # bandwidth
        harmonic_midi = hz_to_midi(self.f0)
        lower = midi_to_hz(harmonic_midi - (self.bw_Q/2))
        upper = midi_to_hz(harmonic_midi + (self.bw_Q/2))

        f0 = self.f0.unsqueeze(0) # (1, n_band)
        low = lower.unsqueeze(0)
        upp = upper.unsqueeze(0)
        fft_bins = self.fft_bins.unsqueeze(1) # (n_bins, 1)

        up_slope = torch.matmul(fft_bins, 1/(f0-low)) - (low / (f0-low))
        down_slope = torch.matmul(fft_bins, -1/(upp-f0)) + (upp / (upp-f0))
        fb = torch.max(self.zero, torch.min(down_slope, up_slope))
        return fb

    def to_device(self, device, n_bins):
        self.f0 = self.f0.to(device)
        self.bw_Q = self.bw_Q.to(device)

        # fft bins
        self.fft_bins = torch.linspace(0, self.sample_rate//2, n_bins)
        self.fft_bins = self.fft_bins.to(device)
        self.zero = torch.zeros(1)
        self.zero = self.zero.to(device)

    def forward(self, waveform):
        # stft
        spectrogram = self.spec(waveform)

        # to device
        self.to_device(waveform.device, spectrogram.size(1))

        # triangle filter
        harmonic_fb = self.get_harmonic_fb()
        harmonic_spec = torch.matmul(spectrogram.transpose(1, 2), harmonic_fb).transpose(1, 2)

        # (batch, channel, length) -> (batch, harmonic, f0, length)
        b, c, l = harmonic_spec.size()
        harmonic_spec = harmonic_spec.view(b, self.n_harmonic, self.level, l)

        # amplitude to db
        harmonic_spec = self.amplitude_to_db(harmonic_spec)
        return harmonic_spec


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class MelSpec(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=512,
                 win_length=None,
                 hop_length=None,
                 pad=0,
                 power=2,
                 normalized=False,
                 n_harmonic=6,
                 semitone_scale=2,
                 bw_alpha=0.1079,
                 bw_beta=24.7,
                 bw_Q=2.0,
                 learn_f0=False,
                 learn_bw=None):
        super(MelSpec, self).__init__()

        # Parameters
        self.sample_rate = sample_rate
        self.n_harmonic = n_harmonic

        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                                         win_length=win_length, hop_length=hop_length,
                                                         pad=pad, n_mels=128, window_fn=torch.hann_window,
                                                         wkwargs=None)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform):
        # stft
        spectrogram = self.spec(waveform)
        melspec = self.amplitude_to_db(spectrogram)
        return melspec



class ResNet_mp(nn.Module):
    def __init__(self, input_channels, conv_channels):
        super(ResNet_mp, self).__init__()
        self.num_class = 50

        # residual convolution
        self.res1 = Conv3_2d(input_channels, conv_channels, 2)
        self.res2 = Conv3_2d_resmp(conv_channels, conv_channels, 2)
        self.res3 = Conv3_2d(conv_channels, conv_channels*2, 2)
        self.res4 = Conv3_2d_resmp(conv_channels*2, conv_channels*2, 2)
        self.res5 = Conv3_2d_resmp(conv_channels*2, conv_channels*2, 2)
        self.res6 = Conv3_2d_resmp(conv_channels*2, conv_channels*2, (2,3))
        self.res7 = Conv3_2d_resmp(conv_channels*2, conv_channels*2, (2,3))

        # fully connected
        self.fc_1 = nn.Linear(conv_channels * 2, conv_channels * 2)
        self.bn = nn.BatchNorm1d(conv_channels * 2)
        self.fc_2 = nn.Linear(conv_channels * 2, self.num_class)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # residual convolution
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = x.squeeze(2)

        # global max pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # fully connected
        x = self.fc_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = self.activation(x)

        return x


class ResNet_3D(nn.Module):
    def __init__(self, input_channels, conv_channels):
        super(ResNet_3D, self).__init__()
        self.num_class = 50

        # residual convolution
        self.res1 = Conv3_3d(1, conv_channels, 2)
        self.res2 = Conv3_2d_resmp(conv_channels, conv_channels, 2)
        self.res3 = Conv3_2d(conv_channels, conv_channels*2, 2)
        self.res4 = Conv3_2d_resmp(conv_channels*2, conv_channels*2, 2)
        self.res5 = Conv3_2d_resmp(conv_channels*2, conv_channels*2, 2)
        self.res6 = Conv3_2d_resmp(conv_channels*2, conv_channels*2, (2,3))
        self.res7 = Conv3_1d_resmp(conv_channels*2, conv_channels*2, 3)

        # fully connected
        self.fc_1 = nn.Linear(conv_channels * 2, conv_channels * 2)
        self.bn = nn.BatchNorm1d(conv_channels * 2)
        self.fc_2 = nn.Linear(conv_channels * 2, self.num_class)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # residual convolution
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = x.squeeze(2)
        x = self.res7(x)

        # global max pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # fully connected
        x = self.fc_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = self.activation(x)

        return x


class Conv3_2d(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(Conv3_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class Conv3_3d(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(Conv3_3d, self).__init__()
        self.conv = nn.Conv3d(1, output_channels, (6,3,3), padding=(0,1,1))
        self.bn = nn.BatchNorm3d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn(self.conv(x)))
        x = x.squeeze(2)
        out = self.mp(x)
        return out


class Conv_3d_resmp(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(Conv3_3d_resmp, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        out = out + x
        out = self.mp(self.relu(out))
        return out


class Conv3_2d_resmp(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(Conv3_2d_resmp, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        out = x + out
        out = self.mp(self.relu(out))
        return out

class Conv3_1d_resmp(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(Conv3_1d_resmp, self).__init__()
        self.conv_1 = nn.Conv1d(input_channels, output_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm1d(output_channels)
        self.conv_2 = nn.Conv1d(output_channels, output_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(pooling)
    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        out = x + out
        out = self.mp(self.relu(out))
        return out

