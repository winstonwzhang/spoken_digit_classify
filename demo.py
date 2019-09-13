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
import threading
import pyaudio
import pickle
import joblib
import torch

import utils
from utils import model
from utils import data


# HYPERPARAMETERS
THRESH = 0.5  # threshold for novelty detection
CHUNK = 4096  # chunk size of input audio stream
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
        self.mdl.load_state_dict(torch.load(state_dict_path))
        self.mdl.eval()

    def predict(self, test_wav):
        """Classify [PRED_TIME] seconds of audio. Returns predicted label."""
        # preprocess audio
        input_img = data.audio_to_mfcc(data.conf, test_wav)
        img = torch.from_numpy(input_img)
        img = img.unsqueeze_(0)  # add singleton dimension
        
        # feed mfcc to model
        output = self.mdl(img)
        _, preds = torch.max(output, 1)
        
        # does pred meet threshold?
        
        
        return preds


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
        self.tape = np.empty(self.rate * self.tapeLength)
        self.tapePred = ''  # prediction for current audio in tape
        self.tapeProb = ''  # prediction probability

        self.state = 'b'  # current state
        self.shutdown = False  # change to true if user presses 'Q'

        self.p = pyaudio.PyAudio()  # start the PyAudio class

        if startStreaming:
            self.stream_start()

    def printBar(self, amt, total, fill='█'):
        """Prints a progress bar."""
        bar = fill * int(30 * amt // total)
        sys.stdout.write(("\r|%s|" % bar).ljust(70))

        if self.state == 'b':
            sys.stdout.write("\n... Loading ...")

        if self.state == 'l':
            sys.stdout.write(
                "\nInput [p] to predict, and [q] to quit.")

        elif self.state == 'p':
            sys.stdout.write(
                "\nPrediction: %s. Prob: %s. Input [q] to cancel." % (
                    self.tapePred, self.tapeProb))

    # // LOWEST LEVEL AUDIO ACCESS
    # pure access to microphone and stream operations

    def stream_read(self):
        """return values for a single chunk"""
        data = np.frombuffer(self.stream.read(self.chunk), dtype=np.int16)
        # data = data[:self.chunk]
        data = data.astype(np.float32)
        # convert to monochannel audio
        data = np.vstack((data[::2] / 2, data[1::2] / 2)).sum(axis=0)
        data = data.astype(np.int16)
        # self.player.write(data, self.chunk)
        return data

    def stream_start(self):
        """connect to the audio device and start a stream"""
        print(" -- stream started")
        self.stream = self.p.open(format=pyaudio.paInt16, channels=2,
                                  rate=self.rate, input=True,
                                  frames_per_buffer=self.chunk)
        self.player = self.p.open(format=pyaudio.paInt16, channels=1,
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
        self.tape[:-self.chunk] = self.tape[self.chunk:]
        self.tape[-self.chunk:] = self.stream_read()

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
                self.printBar(peak, self.rate)
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

        time.sleep(5)  # wait for tape to fill
        self.state = 'l'  # start listening to input

        while not self.shutdown:
            choice = input()

            if choice == 'p' and self.state == 'l':
                # predict the past 5 seconds
                self.state = 'p'
                p_thread = threading.Thread(target=self.predict, args=[])
                p_thread.start()

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

    TISR_obj.close()

    print("DONE")
