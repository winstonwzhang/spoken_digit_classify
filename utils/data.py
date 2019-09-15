import os
import math
import time
import random
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import librosa
import librosa.display
import librosa.effects
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


# default data hyperparameters
class conf:
    sr = 8000  # sampling rate
    duration = 2.3  # how long input samples to our model should be
    hop_length = int(sr * 0.0115)  # step size half of frame
    fmin = 0.0
    fmax = 22050.0  # freq range of human speech
    n_mels = 25  # number of mfcc coefficients
    n_fft = int(sr * 0.023)  # frames of 0.023 sec
    padmode = 'constant'
    samples = sr * duration
    aug_probs = {'noise': 0.7,
                 'pitch': 0.3,
                 'speed': 0.7,
                 'shift': 0.8}  # prob of augment being applied


class Transform:
    '''Audio data augmentation class.'''
    def __init__(self, conf, wav):
        self.wav = wav
        self.conf = conf
    
    def run_transforms(self):
        x = self.wav
        if random.uniform(0, 1) < self.conf.aug_probs['pitch']:
            x = self.pitch_shift(x)
        if random.uniform(0, 1) < self.conf.aug_probs['speed']:
            x = self.speed_change(x)
        return x
    
    def add_noise(self, wav):
        if random.uniform(0, 1) < self.conf.aug_probs['noise']:
            noise = np.random.normal(size=len(wav))
            factor = 0.005 * random.uniform(0, 1) * max(wav)
            return wav + factor * noise
        else:
            return wav
    
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


def pad_audio(conf, audio, transform):
    # pad or cut audio to same length as conf.samples
    if len(audio) > conf.samples:
        audio = audio[0:0+conf.samples]
    elif len(audio) < conf.samples: # pad blank
        padding = conf.samples - len(audio)    # add padding at both ends
        if transform and random.uniform(0, 1) < conf.aug_probs['shift']:
            offset = random.randint(0, padding)
        else:
            offset = math.ceil(padding // 2)
        # random shift of audio (speech not always in center of spectrogram)
        padwidth = (offset, math.ceil(padding - offset))
        audio = np.pad(audio, padwidth, conf.padmode)
    return audio
  
def audio_to_mfcc(conf, audio):
    spectrogram = librosa.feature.mfcc(audio, 
                                       sr=conf.sr,
                                       n_mfcc=conf.n_mels,
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
    # librosa slow to load for some reason
    _, x = scipy.io.wavfile.read(pathname)
    x = np.float32(x)
    # apply audio data augmentations (pitch, speed)
    if transform:
        tr = Transform(conf, x)
        x = tr.run_transforms()
    # padding for uniform input length
    x = pad_audio(conf, x, transform)
    # add noise to simulate realistic conditions
    if transform:
        x = tr.add_noise(x)
    # get mfccs
    mfcc = audio_to_mfcc(conf, x)
    # show_mfcc(conf, mfcc)
    return mfcc

def norm_mfcc(mfcc_data):
    # normalize mfcc coefficients within each sample
    mfcc_mean = np.mean(mfcc_data, axis=1)
    mfcc_std = np.std(mfcc_data, axis=1)
    
    # broadcast across all time frames
    mfcc_mean = mfcc_mean[:, np.newaxis]
    mfcc_std = mfcc_std[:, np.newaxis]
    mfcc_data = (mfcc_data - mfcc_mean) / mfcc_std
    
    return mfcc_data


class Digit_Dataset(Dataset):
    '''Dataset class that pads and converts a sample into mfcc.'''
    def __init__(self, wav_dir, transform=True):
        self.wav_dir = wav_dir
        self.data = os.listdir(wav_dir)
        # keep only .wav files
        self.data = [s for s in self.data if '.wav' in s]
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
            
            mfcc_data = norm_mfcc(mfcc_data)
            
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
