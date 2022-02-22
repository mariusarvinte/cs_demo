Minimalistic code for fast, near-optimal CDL-X channel estimation, showcasing the research perils of synthetic data and serving as a tough-to-beat baseline for deep learning.

The algorithm used is an in-house, optimized version of the ```fsAD``` algorithm from https://arxiv.org/pdf/1709.06832.pdf (no original source code is available).

# Google Colab
(may require you to create a ```data``` folder and place the data file in)

https://colab.research.google.com/drive/1exGfyqZ_sae-63vdqS-NndRRsZ9D136X?usp=sharing

# Requirements
0. Create a new conda environment ```conda create -n fsad``` and activate it ```conda activate fsad```.
1. Install SigPy - https://sigpy.readthedocs.io/en/latest/ - using ```conda install -c frankong sigpy```.
2. Install the remainder of packages with Anaconda: ```numpy```, ```matplotlib```, ```scipy```, ```tqdm```, ```h5py```.

# Instructions
All the code is click-to-run and by default performs channel estimation on CDL-A channels of size ```16 x 64```, with a pilot density of ```0.6```, on a wide range of SNR values.

1. Run ```main.py```.
2. (Optional, if you prefer notebooks) Run ```demo.ipynb```.

# Citations
The original ```fsAD``` paper:
```
@article{zhang2017atomic,
  title={Atomic norm denoising-based joint channel estimation and faulty antenna detection for massive MIMO},
  author={Zhang, Peng and Gan, Lu and Ling, Cong and Sun, Sumei},
  journal={IEEE Transactions on Vehicular Technology},
  volume={67},
  number={2},
  pages={1389--1403},
  year={2017},
  publisher={IEEE}
}
```

SigPy (an amazing package for linear inverse problems with CS, don't let MRI scare you):
```
@inproceedings{ong2019sigpy,
  title={SigPy: a python package for high performance iterative reconstruction},
  author={Ong, Frank and Lustig, Michael},
  booktitle={Proceedings of the ISMRM 27th Annual Meeting, Montreal, Quebec, Canada},
  volume={4819},
  year={2019}
}
```
