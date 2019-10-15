import os
import numpy as np
import pandas as pd
import glob
from essentia.standard import MonoLoader
from sklearn.utils import shuffle
import fire
import tqdm


class Processor:
	def __init__(self):
		self.fs = 16000
		self.classes = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 
				'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 
				'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
				'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
		self.ix_to_name = {i: name for i, name in enumerate(self.classes)}
		self.name_to_ix = {name: i for i, name in enumerate(self.classes)}

	def get_paths(self, data_path):
		audio_paths = glob.glob(os.path.join(data_path, 'keyword', 'raw/*/*.wav'))
		noise_paths = glob.glob(os.path.join(data_path, 'keyword', 'raw/_background_noise_/*.wav'))

		with open(os.path.join(data_path, 'keyword', 'raw/validation_list.txt')) as f:
			val_paths = f.read().splitlines()
			val_paths = [os.path.join(data_path, 'keyword', 'raw', path) for path in val_paths]
		with open(os.path.join(data_path, 'keyword', 'raw/testing_list.txt')) as f:
			test_paths = f.read().splitlines()
			test_paths = [os.path.join(data_path, 'keyword', 'raw', path) for path in test_paths]

		train_paths = list(set(audio_paths) - set(val_paths) - set(test_paths) - set(noise_paths))

		train_paths.sort(), val_paths.sort(), test_paths.sort()
		return train_paths, val_paths, test_paths

	def get_df(self, train_paths, val_paths, test_paths):
		paths = train_paths + val_paths + test_paths
		ids = ['/'.join(p.split('/')[-2:]) for p in paths]
		labels = [_id.split('/')[0] for _id in ids]
		label_num = [self.name_to_ix[_id.split('/')[0]] for _id in ids]
		splits = ['train'] * len(train_paths) + ['val'] * len(val_paths) + ['test'] * len(test_paths)

		df = pd.DataFrame({'id': ids, 'label': labels, 'label_num': label_num, 'split': splits, 'path': paths})
		df = shuffle(df, random_state=123)
		return df

	def get_npy(self, fn):
		loader = MonoLoader(filename=fn, sampleRate=self.fs)
		x = loader()
		return x

	def iterate(self, data_path, df):
		npy_path = os.path.join(data_path, 'keyword', 'npy')
		if not os.path.exists(npy_path):
			os.makedirs(npy_path)
		for _class in self.classes:
			_subpath = os.path.join(npy_path, _class)
			if not os.path.exists(_subpath):
				os.makedirs(_subpath)
		fl = list(df['path'])
		for fn in tqdm.tqdm(fl):
			npy_fn = os.path.join(npy_path, fn.split('/')[-2], fn.split('/')[-1][:-3]+'npy')
			try:
				x = self.get_npy(fn)
				np.save(open(npy_fn, 'wb'), x)
				df.loc[df.path==fn, 'path'] = npy_fn
			except RuntimeError:
				# some audio files are broken
				print(fn)
				continue
		return df

	def run(self, data_path):
		train_paths, val_paths, test_paths = self.get_paths(data_path)
		df = self.get_df(train_paths, val_paths, test_paths)
		df = self.iterate(data_path, df)
		df.to_csv(os.path.join(data_path, 'keyword', 'df.csv'), sep='\t', header=None)
		

if __name__ == '__main__':
	p = Processor()
	fire.Fire({'run': p.run})
