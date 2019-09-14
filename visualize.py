import numpy as np
import scipy
import librosa
import matplotlib.pyplot as plt
import utils
from utils import data

conf = data.conf

# compare two wav files
fa = 'recordings/4_nicolas_10.wav'
fb = 'clips/1568430088.wav'

ma = data.read_as_mfcc(conf, fa, False)
mb = data.read_as_mfcc(conf, fb, False)

# plot mfcc
plt.subplot(2, 1, 1)
librosa.display.specshow(ma,
                         x_axis='time',
                         sr=conf.sr,
                         hop_length=conf.hop_length)
plt.colorbar(format='%+2.0f dB')
plt.title(fa)

plt.subplot(2, 1, 2)
librosa.display.specshow(mb,
                         x_axis='time',
                         sr=conf.sr,
                         hop_length=conf.hop_length)
plt.colorbar(format='%+2.0f dB')
plt.title(fb)

plt.show()