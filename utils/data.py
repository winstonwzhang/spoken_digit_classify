import os
import math
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import librosa
import librosa.display
import librosa.effects
import torch
#import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


# default configuration
class conf:
    sr = 8000
    duration = 2.3 # max length
    hop_length = int(sr * 0.0115) # step size half of frame
    fmin = 0.0
    fmax = 22050.0  # freq range of human speech
    n_mels = 20
    n_fft = int(sr * 0.023)  # frames of 0.023 sec
    padmode = 'constant'
    samples = sr * duration


class Transform:
    '''Audio data augmentation class.'''
    def __init__(self, conf, wav):
        self.wav = wav
        self.conf = conf
    
    def run_transforms(self):
        x = self.wav
        if random.uniform(0, 1) > 0.3:
            x = self.add_noise(x)
        if random.uniform(0, 1) > 0.7:
            x = self.pitch_shift(x)
        if random.uniform(0, 1) > 0.5:
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
        factor = np.random.normal(1, 0.2, 1)
        tmp = librosa.effects.time_stretch(wav, factor)
        minlen = min(wav.shape[0], tmp.shape[0])
        wav += 0
        wav[0:minlen] = tmp[0:minlen]
        return wav


def pad_audio(conf, audio):
    # make it unified length to conf.samples
    if len(audio) > conf.samples:
        audio = audio[0:0+conf.samples]
    elif len(audio) < conf.samples: # pad blank
        padding = conf.samples - len(audio)    # add padding at both ends
        offset = math.ceil(padding // 2)
        padwidth = (offset, math.ceil(conf.samples - len(audio) - offset))
        audio = np.pad(audio, padwidth, conf.padmode)
    return audio
  
def audio_to_mfcc(conf, audio):
    audio = pad_audio(conf, audio)

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
    #x, _ = torchaudio.load(pathname)
    #x = x[0, :].detach().numpy()
    _, x = scipy.io.wavfile.read(pathname)
    x = x.astype(np.float32)
    #x = scipy.ndimage.median_filter(x, 3)
    x = pad_audio(conf, x)
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
        self.labels = [int(s[0]) for s in self.data]
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

            sample = {'img': img, 'label': label}

            return sample
    
    def visualize_labels(self, transform=True):
        for i in range(4):
            idx = self.labels.index(i)
            mfcc = read_as_mfcc(conf, self.wav_dir + self.data[idx], transform)
            plt.subplot(2, 2, i+1)
            librosa.display.specshow(mfcc,
                                     x_axis='time',
                                     sr=conf.sr,
                                     hop_length=conf.hop_length)
            plt.colorbar(format='%+2.0f dB')
            plt.title(str(i))
        plt.tight_layout()
        plt.show()
