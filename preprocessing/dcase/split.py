# coding: utf-8

import os
import glob
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import fire


class Split:
	def __init__(self):
		print('start preprocessing')
		self.classes = ['Train horn', 'Air horn, truck horn', 'Car alarm', 'Reversing beeps', 'Ambulance (siren)',
						'Police car (siren)', 'Fire engine, fire truck (siren)', 'Civil defense siren', 'Screaming', 'Bicycle',
						'Skateboard', 'Car', 'Car passing by', 'Bus', 'Truck', 'Motorcycle', 'Train']
		self.class_to_index = {c: i for i, c in enumerate(self.classes)}

	def make_path(self, fn):
		return os.path.join(self.data_path, 'dcase', fn)

	def get_csv_dataframe(self):
		df_train = pd.read_csv(self.make_path('groundtruth_weak_label_training_set.csv'), delimiter='\t', names=['file', 'start', 'end', 'label'])
		df_test = pd.read_csv(self.make_path('groundtruth_weak_label_testing_set.csv'), delimiter='\t', names=['file', 'start', 'end', 'label'])
		df_eval = pd.read_csv(self.make_path('groundtruth_weak_label_evaluation_set.csv'), delimiter='\t', names=['file', 'start', 'end', 'label'])

		df_train['path'] = [os.path.join(self.data_path, 'dcase', 'npy', 'train', 'Y'+f[:-3]+'npy') for f in df_train['file']]
		df_test['path'] = [os.path.join(self.data_path, 'dcase', 'npy', 'test', 'Y'+f[:-3]+'npy') for f in df_test['file']]
		df_eval['path'] = [os.path.join(self.data_path, 'dcase', 'npy', 'evaluation', 'Y'+f[:-3]+'npy') for f in df_eval['file']]

		df_train = pd.concat([df_train, df_test])
		return df_train, df_eval

	def validation_split(self, df_train, df_eval):
		val_files = []
		for c in self.classes:
			df_class = df_train[df_train['label'] == c]
			val_files += df_class.sample(frac=0.1, random_state=123)['file'].tolist()
		val_files = list(set(val_files))

		is_val = df_train['file'].isin(val_files)
		df_val = df_train[is_val].assign(split='val')
		df_train = df_train[~is_val].assign(split='train')
		df_eval = df_eval.assign(split='test')

		df = pd.concat([df_train, df_val, df_eval])
		return df

	def encoding(self, df):
		label = df.groupby('file')['label'].apply(list)
		label.iloc[:] = [self.encode(l) for l in label]
		label = label.to_frame().reset_index()
		df = df.drop_duplicates('file').drop('label', axis=1).merge(label, on='file')
		return df

	def encode(self, label):
		x = np.zeros(shape=len(self.classes), dtype=np.float32)
		x[[self.class_to_index[l] for l in label]] = 1.
		return x

	def run(self, data_path):
		self.data_path = data_path
		df_train, df_eval = self.get_csv_dataframe()
		df = self.validation_split(df_train, df_eval)
		df = self.encoding(df)
		df = shuffle(df, random_state=123)
		df.to_csv(os.path.join(data_path, 'dcase', 'df.csv'), sep='\t', header=None)
		print('done!')

if __name__ == '__main__':
	s = Split()
	fire.Fire({'run': s.run})
