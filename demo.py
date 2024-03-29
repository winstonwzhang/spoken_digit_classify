"""
Command line program for real-time speech classification.

INPUT:
state_dict - path to trained model's pytorch state_dict .pt file

///// USAGE EXAMPLE:

python demo.py models/fsdd_cnn_sdict.pt

/////

Winston Zhang September 2019.
"""
import os
from os.path import join
import sys
import time
import numpy as np
import scipy
import threading
import pyaudio
import librosa
import pickle
import joblib
import torch

import utils
from utils import model
from utils import data

import warnings
warnings.filterwarnings("ignore")

# HYPERPARAMETERS
THRESH = 0.5  # threshold for novelty detection
CHUNK = 4092  # chunk size of input audio stream
RATE = data.conf.sr  # sampling rate of input audio stream
PRED_TIME = data.conf.duration  # audio time in sec used to predict


class PredModel(object):
    """
    Stores trained Pytorch model and uses it to predict.
    Input: state_dict_path - path to trained model's state_dict.
    """

    def __init__(self, state_dict_path):
        """Read in state_dict and initialize model."""
        self.mdl = model.BaseModel()
        # if cpu-only
        f_obj = torch.load(state_dict_path, map_location=torch.device('cpu'))
        self.mdl.load_state_dict(f_obj)
        self.mdl.eval()

    def predict(self, test_wav):
        """Classify [PRED_TIME] seconds of audio. Returns predicted label."""
        # preprocess audio
        input_img = data.audio_to_mfcc(data.conf, test_wav)
        input_img = data.norm_mfcc(input_img)
        
        class_preds = model.model_predict(input_img, self.mdl)
        # does pred meet threshold?
        pred = np.argmax(class_preds, axis=1)
        prob = class_preds[:, pred]
        
        return str(pred), str(prob)


class RealTimeRecord(object):
    """
    The RealTimeRecord class saves the previous [PRED_TIME] seconds
    of sound and continuously predicts on the stored audio.
    """

    def __init__(self, mdl=None, startStreaming=True):
        """fire up the class."""
        print(" -- initializing record object")

        self.chunk = CHUNK  # number of data points to read at a time
        self.rate = RATE  # time resolution of the recording device (Hz)

        self.clf = mdl  # PredModel class

        # for tape recording (continuous "tape" of recent audio)
        self.tapeLength = PRED_TIME  # seconds
        self.tape = np.empty(int(self.rate * self.tapeLength))
        self.tapePred = ''  # prediction for current audio in tape
        self.tapeProb = ''  # prediction probability

        self.state = 'b'  # current state
        self.shutdown = False  # change to true if user presses 'Q'

        self.p = pyaudio.PyAudio()  # start the PyAudio class

        if startStreaming:
            self.stream_start()

    def printBar(self, peak_ratio, fill='='):
        """Prints a progress bar."""
        if peak_ratio > 1:
            peak_ratio = 1
        bar_length = 30
        bar = fill * int(29 * peak_ratio)
        spaces = ' ' * (bar_length - len(bar))

        if self.state == 'b':
            msg = "... Loading ..."

        if self.state == 'l':
            msg = "Input [p] to predict, [s] to save audio clip, and [q] to quit."

        elif self.state == 'p':
            msg = "Prediction: {0}. Prob: {1}. Input [q] to cancel.".format(
                self.tapePred, self.tapeProb)
        
        elif self.state == 's':
            msg = "Last {0} seconds saved to clips/".format(PRED_TIME)
        
        sys.stdout.write(("\r|{0}|{1}".format(bar + spaces, msg)).ljust(70))

    # // LOWEST LEVEL AUDIO ACCESS
    # pure access to microphone and stream operations

    def stream_read(self):
        """return values for a single chunk"""
        dat = np.fromstring(self.stream.read(self.chunk), dtype=np.float32)
        #import pdb; pdb.set_trace()
        dat = np.vstack((dat[::2] / 2, dat[1::2] / 2)).sum(axis=0)
        dat = scipy.ndimage.median_filter(dat, 3)  # filter noise
        self.player.write(dat, self.chunk)
        return dat

    def stream_start(self):
        """connect to the audio device and start a stream"""
        print(" -- stream started")
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=2,
                                  rate=self.rate, input=True,
                                  frames_per_buffer=self.chunk)
        self.player = self.p.open(format=pyaudio.paFloat32, channels=1,
                                  rate=self.rate, output=True,
                                  frames_per_buffer=self.chunk)

    def stream_stop(self):
        """close the stream but keep the PyAudio instance alive."""
        if 'stream' in locals():
            self.stream.stop_stream()
            self.stream.close()
        print(" -- stream CLOSED")

    def close(self):
        """gently detach from things."""
        self.stream_stop()
        self.p.terminate()

    # // TAPE METHODS
    # tape is previous [PRED_TIME] sec of audio continuously recorded.
    # self.tape contains this data.
    # newest data is always at the end. Don't modify data on the tape,
    # but rather analyze as you read from it.

    def tape_add(self):
        """add a single chunk to the tape."""
        new_chunk = self.stream_read()
        self.tape[:-self.chunk] = self.tape[self.chunk:]
        self.tape[-self.chunk:] = new_chunk

    def tape_flush(self):
        """completely fill tape with new data."""
        readsInTape = int(self.rate * self.tapeLength / self.chunk)
        print(" -- flushing %d s tape with %dx%.2f ms reads" %
              (self.tapeLength, readsInTape, self.chunk / self.rate))
        for i in range(readsInTape):
            self.tape_add()

    def tape_forever(self):
        """Keep getting new audio and updating tape."""
        try:
            while not self.shutdown:
                self.tape_add()
                # for length of cmd line bar
                peak = int(np.average(np.abs(self.tape)))
                #print(self.tape)
                self.printBar(peak / 1000)
                time.sleep(0.25)
        except Exception as e:
            print(e)
            return

    # // ACTION METHODS
    # functions for audio prediction.

    def predict(self):
        """Predict speaker of the last five seconds stored in tape."""
        while self.state != 'l':
            # predict on stored tape audio
            time.sleep(0.25)
            self.tapePred, self.tapeProb = self.clf.predict(self.tape)

    def listen(self):
        """Begin audio recording and prediction."""
        voicebar = threading.Thread(target=self.tape_forever, args=[])
        voicebar.start()

        time.sleep(3)  # wait for tape to fill
        self.state = 'l'  # start listening to input

        while not self.shutdown:
            choice = input()

            if choice == 'p' and self.state == 'l':
                # predict the past 5 seconds
                self.state = 'p'
                p_thread = threading.Thread(target=self.predict, args=[])
                p_thread.start()
            
            elif choice == 's' and self.state == 'l':
                # save currently stored clip
                self.state = 's'
                scipy.io.wavfile.write(
                    'clips/' + str(int(time.time())) + '.wav', self.rate, self.tape)
                time.sleep(3)
                self.state = 'l'

            elif choice == 'q':
                # quit or cancel action
                if self.state == 'l':
                    self.shutdown = True
                    break
                elif self.state == 'p':
                    self.state = 'l'

            # don't continuously listen to input
            time.sleep(0.1)


if __name__ == "__main__":
    """Command line error checking."""

    if len(sys.argv) != 2:
        print("USAGE: demo.py [state_dict_path]")
        exit()
    sdict_path = sys.argv[1]

    # path error checking
    sdict_path = os.path.normpath(sdict_path)
    if not os.path.isfile(sdict_path):
        print("input path does not exist")
        exit()

    # initalize speaker database
    mdl_obj = PredModel(sdict_path)

    # start recording audio
    RTR_obj = RealTimeRecord(mdl=mdl_obj)
    RTR_obj.listen()

    RTR_obj.close()

    print("DONE")
