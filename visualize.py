import numpy as np
import scipy
import librosa
import matplotlib.pyplot as plt
import torch

import utils
from utils import data
from utils import model

import warnings
warnings.filterwarnings("ignore")

conf = data.conf

# compare two wav files
fa = 'recordings/1_jackson_40.wav'
fb = 'clips/1568531472.wav'
state_dict_path = 'models/fsdd_cnn_sdict.pt'

# compare amplitudes
sra, wa = scipy.io.wavfile.read(fa)
srb, wb = scipy.io.wavfile.read(fb)
print('sr A: {}, sr B: {}'.format(sra, srb))
print('mean amp A: {}, mean amp B: {}'.format(np.mean(wa), np.mean(wb)))

ma = data.read_as_mfcc(conf, fa, False)
mb = data.read_as_mfcc(conf, fb, False)

ma = data.norm_mfcc(ma)
mb = data.norm_mfcc(mb)

# get predictions
mdl = model.BaseModel()
f_obj = torch.load(state_dict_path, map_location=torch.device('cpu'))
mdl.load_state_dict(f_obj)
mdl.eval()

aclass_preds = model.model_predict(ma, mdl)
apred = np.argmax(aclass_preds, axis=1)
aprob = aclass_preds[:, apred]

bclass_preds = model.model_predict(mb, mdl)
bpred = np.argmax(bclass_preds, axis=1)
bprob = bclass_preds[:, bpred]
print('apred: {}, aprobs: {}'.format(apred, aclass_preds))
print('bpred: {}, bprobs: {}'.format(bpred, bclass_preds))

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