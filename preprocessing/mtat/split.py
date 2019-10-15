# coding: utf-8

import os
import glob
import numpy as np
import pandas as pd
import fire


class Split:
	def __init__(self):
		print('start preprocessing')

	def get_csv_dataframe(self):
		self.csv_df = pd.read_csv(os.path.join(self.data_path, 'mtat', 'annotations_final.csv'), header=None, index_col=None, sep='\t')

	def get_top50_tags(self):
		tags = list(self.csv_df.loc[0][1:-1])
		tag_count = [np.array(self.csv_df[i][1:], dtype=int).sum() for i in range(1, 189)]
		top_50_tag_index = np.argsort(tag_count)[::-1][:50]
		top_50_tags = np.array(tags)[top_50_tag_index]
		np.save(open(os.path.join(self.data_path, 'mtat', 'tags.npy'), 'wb'), top_50_tags)
		return top_50_tag_index

	def write_tags(self, top_50_tag_index):
		binary = np.zeros((25863, 50))
		titles = []
		idx = 0
		for i in range(1, 25864):
			features = np.array(self.csv_df.loc[i][top_50_tag_index+1], dtype=int)
			title = self.csv_df.loc[i][189]
			#if np.sum(features) != 0:
			binary[idx] = features
			idx += 1
			titles.append(title)

		binary = binary[:len(titles)]
		np.save(open(os.path.join(self.data_path, 'mtat', 'binary.npy'), 'wb'), binary)
		return titles, binary

	def split(self, titles, binary):
		tr = []
		val = []
		test = []
		for i, title in enumerate(titles):
			if int(title[0], 16) < 12:
				if binary[i].sum() > 0:
					tr.append(str(i)+'\t'+title)
			elif int(title[0], 16) < 13:
				if binary[i].sum() > 0:
					val.append(str(i)+'\t'+title)
			else:
				if binary[i].sum() > 0:
					test.append(str(i)+'\t'+title)
		self.get_exist(tr, val, test)

	def get_exist(self, tr, val, test):
		tr_exist = []
		val_exist = []
		test_exist = []
		for fn in tr:
			_path = os.path.join(self.data_path, 'mtat', 'npy', fn.split('/')[-1][:-3]+'npy')
			if os.path.exists(_path):
				tr_exist.append(fn)
		for fn in val:
			_path = os.path.join(self.data_path, 'mtat', 'npy', fn.split('/')[-1][:-3]+'npy')
			if os.path.exists(_path):
				val_exist.append(fn)
		for fn in test:
			_path = os.path.join(self.data_path, 'mtat', 'npy', fn.split('/')[-1][:-3]+'npy')
			if os.path.exists(_path):
				test_exist.append(fn)
		np.save(open(os.path.join(self.data_path, 'mtat', 'train.npy'), 'wb'), tr_exist)
		np.save(open(os.path.join(self.data_path, 'mtat', 'valid.npy'), 'wb'), val_exist)
		np.save(open(os.path.join(self.data_path, 'mtat', 'test.npy'), 'wb'), test_exist)

	def run(self, data_path):
		self.data_path = data_path
		self.get_csv_dataframe()
		top_50_tag_index = self.get_top50_tags()
		titles, binary = self.write_tags(top_50_tag_index)
		self.split(titles, binary)


if __name__ == '__main__':
	s = Split()
	fire.Fire({'run': s.run})
