import os
import math
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import librosa.effects
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pdb


# default configuration
class conf:
    sr = 8000
    duration = 2.3 # max length
    hop_length = int(sr * 0.0115) # step size half of frame
    fmin = 0.0
    fmax = 22050.0  # freq range of human speech
    n_mels = 50
    n_fft = int(sr * 0.023)  # frames of 0.023 sec
    padmode = 'constant'
    samples = sr * duration


class Transform:
    '''Audio data augmentation class.'''
    def __init__(self, conf, wav):
        self.wav = wav
        self.conf = conf
    
    def run_transforms(self):
        x = self.add_noise(self.wav)
        x = self.pitch_shift(x)
        x = self.speed_change(x)
        return x
    
    def add_noise(self, wav):
        noise = np.random.normal(size=len(wav))
        factor = 0.005 * random.uniform(0, 1) * max(wav)
        return wav + factor * noise
    
    def pitch_shift(self, wav):
        pitch_pm = 2
        pitch_change = pitch_pm * 2 * random.uniform(0, 1)
        return librosa.effects.pitch_shift(
            wav, self.conf.sr, n_steps=pitch_change)
    
    def speed_change(self, wav):
        factor = random.uniform(0.9, 1.1)
        tmp = librosa.effects.time_stretch(wav, factor)
        minlen = min(wav.shape[0], tmp.shape[0])
        wav += 0
        wav[0:minlen] = tmp[0:minlen]
        return wav


def audio_to_mfcc(conf, audio):
    # make it unified length to conf.samples
    if len(audio) > conf.samples:
        audio = audio[0:0+conf.samples]
    elif len(audio) < conf.samples: # pad blank
        padding = conf.samples - len(audio)    # add padding at both ends
        offset = math.ceil(padding // 2)
        padwidth = (offset, math.ceil(conf.samples - len(audio) - offset))
        audio = np.pad(audio, padwidth, conf.padmode)

    spectrogram = librosa.feature.mfcc(audio, 
                                       sr=conf.sr,
                                       n_mels=conf.n_mels,
                                       hop_length=conf.hop_length,
                                       n_fft=conf.n_fft,
                                       fmin=conf.fmin,
                                       fmax=conf.fmax)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def show_mfcc(conf, mels, title='MFCC'):
    librosa.display.specshow(mels,
                             x_axis='time',
                             sr=conf.sr,
                             hop_length=conf.hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def read_as_mfcc(conf, pathname, transform):
    x, _ = librosa.load(pathname, sr=conf.sr)
    # apply data augmentations
    if transform:
        tr = Transform(conf, x)
        x = tr.run_transforms()
    mels = audio_to_mfcc(conf, x)
    # show_mfcc(conf, mels)
    return mels


class Digit_Dataset(Dataset):
    '''Dataset class that pads and converts a sample into mfcc.'''
    def __init__(self, wav_dir, transform=True):
        self.wav_dir = wav_dir
        self.data = os.listdir(wav_dir)
        self.labels = [s[0] for s in self.data]
        self.transform = transform
          
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with torch.no_grad():
            wav_name = self.data[idx]
            label = self.labels[idx]

            mfcc_data = read_as_mfcc(
                conf, self.wav_dir + wav_name, self.transform)
          
            img = torch.from_numpy(mfcc_data)
            img = img.unsqueeze_(0) # add singleton dimension

            sample = {'img': img, 'label':int(label)}

            return sample
