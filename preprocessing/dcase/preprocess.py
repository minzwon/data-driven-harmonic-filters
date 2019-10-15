import os
import numpy as np
import glob
from essentia.standard import MonoLoader
import fire
import tqdm


class Processor:
    def __init__(self):
        self.fs = 16000

    def get_paths(self, data_path, subset):
        if subset == 'train':
            subdir = 'unbalanced_train_segments_training_set_audio_formatted_and_segmented_downloads'
        elif subset == 'test':
            subdir = 'unbalanced_train_segments_testing_set_audio_formatted_and_segmented_downloads'
        elif subset == 'evaluation':
            subdir = 'evaluation_set_formatted_audio_segments'
        audio_path = os.path.join(data_path, 'dcase', 'raw', subdir)
        self.files = glob.glob(os.path.join(audio_path, '*.wav'))
        self.npy_path = os.path.join(data_path, 'dcase', 'npy', subset)
        if not os.path.exists(self.npy_path):
            os.makedirs(self.npy_path)

    def get_npy(self, fn):
        loader = MonoLoader(filename=fn, sampleRate=self.fs)
        x = loader()
        return x

    def iterate(self):
        for fn in tqdm.tqdm(self.files):
            npy_fn = os.path.join(self.npy_path, fn.split('/')[-1][:-3]+'npy')
            if not os.path.exists(npy_fn):
                try:
                    x = self.get_npy(fn)
                    np.save(open(npy_fn, 'wb'), x)
                except RuntimeError:
                    # some audio files are broken
                    print(fn)
                    continue

    def run(self, data_path):
        self.get_paths(data_path, 'train')
        self.iterate()
        self.get_paths(data_path, 'test')
        self.iterate()
        self.get_paths(data_path, 'evaluation')
        self.iterate()
        

if __name__ == '__main__':
    p = Processor()
    fire.Fire({'run': p.run})
