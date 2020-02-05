# coding: utf-8
import os
import numpy as np
import pandas as pd
from torch.utils import data

class AudioFolder(data.Dataset):
	def __init__(self, root, input_length=None, tr_val='train'):
		df = pd.read_csv(os.path.join(root, 'keyword', 'df.csv'), delimiter='\t', names=['id', 'label', 'label_num', 'split', 'path'])
		df = df[df['split']==tr_val]
		self.tr_val = tr_val
		self.root = root
		self.input_length = input_length
		self.fl = list(df['path'])
		self.gt = list(df['label_num'])

	def __getitem__(self, index):
		npy, tag = self.get_npy(index)
		return npy.astype('float32'), tag

	def get_npy(self, index):
		fn = self.fl[index]
		npy = np.load(fn)
		if len(npy) < self.input_length:
			nnpy = np.zeros(self.input_length)
			ri = int(np.floor(np.random.random(1) * (self.input_length - len(npy))))
			nnpy[ri:ri+len(npy)] = npy
			npy = nnpy
		tag = self.gt[index]
		return npy, tag

	def __len__(self):
		return len(self.fl)


def get_audio_loader(root, batch_size, num_workers=0, input_length=None, tr_val='train'):
	data_loader = data.DataLoader(dataset=AudioFolder(root, input_length=input_length, tr_val=tr_val),
								  batch_size=batch_size,
								  shuffle=True,
								  drop_last=True,
								  num_workers=num_workers)
	return data_loader

