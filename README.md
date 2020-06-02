# Data-driven Harmonic Filters for Audio Representation Learning

For more readable code, please check [this repository](https://github.com/minzwon/sota-music-tagging-models).

## Reference
**Data-driven Harmonic Filters for Audio Representation Learning**, ICASSP 2020 [[pdf](https://ccrma.stanford.edu/~urinieto/MARL/publications/ICASSP2020_Won.pdf)]

-- Minz Won, Sanghyuk Chun, Oriol Nieto, and Xavier Serra

**TL;DR**

- We introduce a stacked band-pass filters. Filters are stacked through channels and their center frequencies are in harmonic relationship, e.g., If the k-th filter in the first channel has a center frequency of 440Hz, k-th filter in the second channel is automatically 880Hz, and the k-th filter in third channel is 1320Hz.
- Center frequencies and bandwidths are learnable.
- Then we simply applied 3x3 CNN.
- It showed SOTA performances in music tagging, keyword spotting, and acoustic event detection tasks.

## Citation
```
@inproceedings{won2020data,
  title={Data-driven harmonic filters for audio representation learning},
  author={Won, Minz and Chun, Sanghyuk and Nieto, Oriol and Serra, Xavier},
  booktitle={Proc. of International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={536--540},
  year={2020},
  organization={IEEE}
}
```