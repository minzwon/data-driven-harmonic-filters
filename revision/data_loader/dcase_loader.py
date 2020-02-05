# coding: utf-8
import os
import numpy as np
import pandas as pd
from torch.utils import data

class AudioFolder(data.Dataset):
	def __init__(self, root, input_length=None):
		df = pd.read_csv(os.path.join(root, 'dcase', 'df.csv'), delimiter='\t', names=['file', 'start', 'end', 'path', 'split', 'label'])
		df = df[df['split']=='train']
		self.root = root
		self.input_length = input_length
		self.fl = list(df['path'])
		self.binary = list(df['label'])

	def __getitem__(self, index):
		npy, tag_binary = self.get_npy(index)
		return npy.astype('float32'), tag_binary.astype('float32')

	def get_npy(self, index):
		fn = self.fl[index]
		npy = np.load(fn, mmap_mode='r')
		if len(npy) < self.input_length:
			nnpy = np.zeros(self.input_length)
			ri = int(np.floor(np.random.random(1) * (self.input_length - len(npy))))
			nnpy[ri:ri+len(npy)] = npy
			npy = nnpy
		random_idx = int(np.floor(np.random.random(1) * (len(npy)-self.input_length)))
		npy = np.array(npy[random_idx:random_idx+self.input_length])
		tag_binary = np.fromstring(self.binary[index][1:-1], dtype=np.float32, sep=' ')
		return npy, tag_binary

	def __len__(self):
		return len(self.fl)


def get_audio_loader(root, batch_size, num_workers=0, input_length=None):
	data_loader = data.DataLoader(dataset=AudioFolder(root, input_length=input_length),
								  batch_size=batch_size,
								  shuffle=True,
								  drop_last=True,
								  num_workers=num_workers)
	return data_loader

