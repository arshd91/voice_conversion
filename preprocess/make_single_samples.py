import sys
sys.path.append('../')
from utils import Sampler
import h5py
import numpy as np
import json
import argparse

'''
    Params in accordance to paper: https://arxiv.org/abs/1804.02812
'''
max_step=5
seg_len=128
mel_band=80
lin_band=513
n_samples=2000000
dset='train'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="""\n--dataset_path\t\th5py file used as data feeder for training phase.
                                              \n--index_output\t\tDestination of json file of sampled indices used \
                                              during training.""")
    parser.add_argument('--dataset_path', default="./vctk_dataset")
    parser.add_argument('--index_output', default="./vctk_dataset_index.json")
    args = parser.parse_args()
    sampler = Sampler(args.dataset_path, max_step=max_step, seg_len=seg_len, dset=dset)
    samples = [sampler.sample_single()._asdict() for _ in range(n_samples)]
    with open(args.index_output, 'w') as f_json:
        json.dump(samples, f_json, indent=4, separators=(',', ': '))





