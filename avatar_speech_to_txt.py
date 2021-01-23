import argparse
import wave
# import numpy as np
from deepspeech import Model


def main():
    parser = argparse.ArgumentParser(description='Audio-to-text translation')
    parser.add_argument('--input', required=True,
                        help='Path to the input audio file (.wav)')
    parser.add_argument('--output', required=True,
                        help='Path to the output text file (.txt)')
    args = parser.parse_args()

    print('Reading audio file')
    input_audio = wave.open(args.input, 'rb')

    print('Loading model')
    # !!! Define model file here !!!
    ds = Model('deepspeech-0.9.3-models.pbmm')
    desired_sample_rate = ds.sampleRate()


if __name__ == '__main__':
    main()
