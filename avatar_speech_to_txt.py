import argparse
import numpy as np
import librosa as lb
from deepspeech import Model


def main():
    parser = argparse.ArgumentParser(description='Audio-to-text translation')
    parser.add_argument('--input', required=True,
                        help='Path to the input audio file (.wav)')
    parser.add_argument('--output', required=True,
                        help='Path to the output text file (.txt)')
    args = parser.parse_args()

    print('Loading model')
    ds = Model('deepspeech-{}-models.pbmm'.format('0.9.3'))  # <- define model version here
    desired_sample_rate = ds.sampleRate()

    print('Loading scorer')
    ds.enableExternalScorer('deepspeech-{}-models.scorer'.format('0.9.3'))  # <- define scorer version here

    print('Reading audio file')
    audio, audio_sample_rate = lb.load(args.input, sr=None)

    if audio_sample_rate != desired_sample_rate:
        print('Warning: input audio sample rate ({}) is different than the desired sampling rate ({}). '
              'Resampling might affect speech recognition.'.format(audio_sample_rate, desired_sample_rate))
        audio = lb.resample(audio, audio_sample_rate, desired_sample_rate)

    # Convert audio from float32 to int16
    audio = (audio*32768).astype(np.int16)

    # Perform speech to text translation
    print('Output: {}'.format(ds.stt(audio)))


if __name__ == '__main__':
    main()
