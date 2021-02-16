import argparse
import numpy as np
import librosa as lb
from deepspeech import Model
import sys
import signal
import time
import os


def extension_check(file, extension, file_use):
    if not file.endswith(extension):
        raise argparse.ArgumentTypeError('{} file must be of type {}'.format(file_use, extension))
    return file


def main():
    parser = argparse.ArgumentParser(description='Translate speech to text and save text to file')
    parser.add_argument('--i', metavar='INPUT', required=True,
                        help='Path to the input audio file (.wav)',
                        type=lambda f: extension_check(f, '.wav', 'Input audio'))
    parser.add_argument('--o', metavar='OUTPUT', required=True,
                        help='Path to the output text file (.txt)',
                        type=lambda f: extension_check(f, '.txt', 'Output'))

    args = parser.parse_args()

    # file check
    if os.path.isfile(args.i):
        print('Reading audio file')
        audio, audio_sample_rate = lb.load(args.i, sr=None)
    else:
        print("Input audio file not found, exiting")
        return

    start_time = time.time()
    print('Loading model')
    ds = Model('deepspeech-{}-models.pbmm'.format('0.9.3'))  # <- define model version here
    desired_sample_rate = ds.sampleRate()
    print('Model loaded in', str(np.round((time.time() - start_time),3)), 'seconds')

    start_time = time.time()
    print('Loading scorer')
    ds.enableExternalScorer('deepspeech-{}-models.scorer'.format('0.9.3'))  # <-define scorer version
    print('Scorer loaded in', str(np.round((time.time() - start_time),3)), 'seconds')

    if audio_sample_rate != desired_sample_rate:
        print('Warning: input audio sample rate ({}) is different than the desired sampling rate ({}). '
              'Resampling might affect speech recognition.'.format(audio_sample_rate, desired_sample_rate))
        audio = lb.resample(audio, audio_sample_rate, desired_sample_rate)

    # DeepSpeech requires the audio to be type int16
    audio = (audio*32768).astype(np.int16)

    start_time = time.time()
    text = ds.stt(audio)
    print('Transcription finished in', str(np.round((time.time() - start_time),3)), 'seconds')

    # Write translation to output file
    with open(args.o, 'w') as file:
        file.write(text)


if __name__ == '__main__':
    main()
