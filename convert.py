import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from utils import Hps
from utils import DataLoader
from utils import Logger
from utils import myDataset
from utils import Indexer
from solver import Solver
from preprocess.tacotron.norm_utils import spectrogram2wav
#from preprocess.tacotron.audio import inv_spectrogram, save_wav
from scipy.io.wavfile import write
#from preprocess.tacotron.mcep import mc2wav
import h5py 
import os 
import soundfile as sf
import pysptk
import pyworld as pw
import sys
import argparse

def sp2wav(sp): 
    #exp_sp = np.exp(sp)
    exp_sp = sp
    wav_data = spectrogram2wav(exp_sp)
    return wav_data

def get_world_param(f_h5, src_speaker, utt_id, tar_speaker, tar_speaker_id, solver, dset='test', gen=True):
    mc = f_h5[f'{dset}/{src_speaker}/{utt_id}/norm_mc'][()]
    converted_mc = convert_mc(mc, tar_speaker_id, solver, gen=gen)
    #converted_mc = mc
    mc_mean = f_h5[f'train/{tar_speaker}'].attrs['mc_mean']
    mc_std = f_h5[f'train/{tar_speaker}'].attrs['mc_std']
    converted_mc = converted_mc * mc_std + mc_mean
    log_f0 = f_h5[f'{dset}/{src_speaker}/{utt_id}/log_f0'][()]
    src_mean = f_h5[f'train/{src_speaker}'].attrs['f0_mean']
    src_std = f_h5[f'train/{src_speaker}'].attrs['f0_std']
    tar_mean = f_h5[f'train/{tar_speaker}'].attrs['f0_mean']
    tar_std = f_h5[f'train/{tar_speaker}'].attrs['f0_std']
    index = np.where(log_f0 > 1e-10)[0]
    log_f0[index] = (log_f0[index] - src_mean) * tar_std / src_std + tar_mean
    log_f0[index] = np.exp(log_f0[index])
    f0 = log_f0
    ap = f_h5[f'{dset}/{src_speaker}/{utt_id}/ap'][()]
    converted_mc = converted_mc[:ap.shape[0]]
    sp = pysptk.conversion.mc2sp(converted_mc, alpha=0.41, fftlen=1024)
    return f0, sp, ap

def synthesis(f0, sp, ap, sr=16000):
    y = pw.synthesize(
            f0.astype(np.float64),
            sp.astype(np.float64),
            ap.astype(np.float64), 
            sr, 
            pw.default_frame_period)
    return y

def convert_sp(sp, c, solver, gen=True):
    c_var = Variable(torch.from_numpy(np.array([c]))).cuda()
    sp_tensor = torch.from_numpy(np.expand_dims(sp, axis=0))
    sp_tensor = sp_tensor.type(torch.FloatTensor)
    converted_sp = solver.test_step(sp_tensor, c_var, gen=gen)
    converted_sp = converted_sp.squeeze(axis=0).transpose((1, 0))
    return converted_sp

def convert_mc(mc, c, solver, gen=True):
    c_var = Variable(torch.from_numpy(np.array([c]))).cuda()
    mc_tensor = torch.from_numpy(np.expand_dims(mc, axis=0))
    mc_tensor = mc_tensor.type(torch.FloatTensor)
    converted_mc = solver.test_step(mc_tensor, c_var, gen=gen)
    converted_mc = converted_mc.squeeze(axis=0).transpose((1, 0))
    return converted_mc

def get_model(hps_path='./hps/vctk.json', model_path='/storage/model/voice_conversion/vctk/clf/model.pkl-109999'):
    hps = Hps()
    hps.load(hps_path)
    hps_tuple = hps.get_tuple()
    solver = Solver(hps_tuple, None)
    solver.load_model(model_path)
    return solver

def convert_all_sp(h5_path, src_speaker, tar_speaker, gen=True, 
        dset='test', speaker_used_path='./vctk_20_speakers_used.txt',
        root_dir='/storage/result/voice_conversion/vctk/p226_to_p225/',
        model_path='/storage/model/voice_conversion/vctk/clf/wo_tanh/model_0.001.pkl-79999'):
    # read speaker id file
    src_speaker = src_speaker[:-1]
    tar_speaker = tar_speaker[:-1]
    with open(speaker_used_path) as f:
        speakers = [line.strip() for line in f]
        speaker2gender = {speaker[:-1]:speaker[-1] for speaker in speakers}
        speaker2id = {speaker[:-1]:i for i, speaker in enumerate(speakers)}
    solver = get_model(hps_path='hps/vctk.json',
            model_path=model_path)
    with h5py.File(h5_path, 'r') as f_h5:
        for utt_id in f_h5[f'{dset}/{src_speaker}']:
            sp = f_h5[f'{dset}/{src_speaker}/{utt_id}/lin'][()]
            converted_sp = convert_sp(sp, speaker2id[tar_speaker], solver, gen=gen)
            wav_data = sp2wav(converted_sp)
            wav_path = os.path.join(root_dir, f'{src_speaker}{speaker2gender[src_speaker]}_{tar_speaker}{speaker2gender[tar_speaker]}_{utt_id}.wav')
            sf.write(wav_path, wav_data, 16000, 'PCM_24')

def convert_all_mc(h5_path, src_speaker, tar_speaker, gen=False, 
        dset='test', speaker_used_path='/storage/feature/voice_conversion/vctk/mcep/en_speaker_used.txt',
        root_dir='/storage/result/voice_conversion/vctk/p226_to_p225',
        model_path='/storage/model/voice_conversion/vctk/clf/wo_tanh/model_0.001.pkl-79999'):
    # read speaker id file
    with open(speaker_used_path) as f:
        speakers = [line.strip() for line in f]
        speaker2id = {speaker:i for i, speaker in enumerate(speakers)}
    solver = get_model(hps_path='hps/vctk.json', 
            model_path=model_path)
    with h5py.File(h5_path, 'r') as f_h5:
        for utt_id in f_h5[f'{dset}/{src_speaker}']:
            f0, sp, ap = get_world_param(f_h5, src_speaker, utt_id, tar_speaker, tar_speaker_id=speaker2id[tar_speaker], solver=solver, dset='test', gen=gen)
            wav_data = synthesis(f0, sp, ap)
            wav_path = os.path.join(root_dir, f'{src_speaker}_{tar_speaker}_{utt_id}.wav')
            sf.write(wav_path, wav_data, 16000, 'PCM_24')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage="""\n--results_dir\t\tDestination folder for converted speech.
                                              \n--dataset_path\t\th5py file used as data feeder for training phase.
                                              \n--model_path\t\tPath to model file.
                                              \n--speakers_used\t\tSpeakers list generated along with h5py file.""")
    parser.add_argument('--results_dir', default="./converted_utt",)
    parser.add_argument('--dataset_path', default="./vctk_20.h5")
    parser.add_argument('--model_path', default="./.model_single_sample.pkl-1")
    parser.add_argument('--speakers_used', default="./speakers_used.txt")
    parser.add_argument('--one', default=False)
    parser.add_argument('--all', default=True)
    parser.add_argument('--source', default=True)
    parser.add_argument('--target', default=True)
    args = parser.parse_args()

    root_dir = args.results_dir
    h5_path = args.dataset_path
    model_path = args.model_path
    speaker_used_path = args.speakers_used
    one_pair = True if args.one == "true" or args.one == "True" else False
    all_pairs = True if args.all == "true" or args.all == "True" else False
    sourceSpeaker = args.source
    targetSpeaker = args.target
    speakers = []
    with open(speaker_used_path) as f:
        speakers = [line.strip() for line in f]

    speakers_wo_gender = [speaker[:-1] for speaker in speakers]
    speaker2gender = {speaker[:-1]:speaker[-1] for speaker in speakers}

    if all_pairs is True:
        for speaker_A in speakers:
           for speaker_B in speakers:
               if speaker_A == speaker_B:
                   continue
               else:
                   dir_path = os.path.join(root_dir, f'p{speaker_A}_p{speaker_B}')
                   if not os.path.exists(dir_path):
                       os.makedirs(dir_path)
                   print(f'Generating converted audio from p{speaker_A} to p{speaker_B}')
                   convert_all_sp(h5_path,speaker_A,speaker_B,
                           root_dir=dir_path,
                           gen=True, model_path=model_path, speaker_used_path=speaker_used_path)
    elif one_pair is True and sourceSpeaker is not "" and targetSpeaker is not "":
        if sourceSpeaker not in speakers_wo_gender and targetSpeaker not in speakers_wo_gender:
            print("Source and Target not in seen data during training.....\nAborting....")
            exit(-1)
        dir_path = os.path.join(root_dir, f'p{sourceSpeaker}{speaker2gender[sourceSpeaker]}_p{targetSpeaker}{speaker2gender[targetSpeaker]}')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        print(f'Generating converted audio from p{sourceSpeaker}{speaker2gender[sourceSpeaker]}_p{targetSpeaker}{speaker2gender[targetSpeaker]}')
        speaker_A = sourceSpeaker+speaker2gender[sourceSpeaker]
        speaker_B = targetSpeaker+speaker2gender[targetSpeaker]
        convert_all_sp(h5_path, speaker_A, speaker_B,
                       root_dir=dir_path,
                       gen=True, model_path=model_path)


