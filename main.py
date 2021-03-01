import os
import argparse
import time
import stt_nemo
import stt_deepspeech
import numpy as np


DEFAULT_INPUT_PATH = '/home/avatar/integration/stt_input.wav'
DEFAULT_OUTPUT_PATH = '/home/avatar/integration/stt_output.txt'

QUARTZNET_MODEL_PATH = 'QuartzNet15x5Base-En.nemo'
JASPER_MODEL_PATH = 'Jasper10x5Dr-En.nemo'
DEEPSPEECH_MODEL_PATH = 'deepspeech-0.9.3-models.pbmm'
DEEPSPEECH_SCORER_PATH = 'deepspeech-0.9.3-models.scorer'

EVALUATION_DATASET_PATHS = ['LibriSpeech/test-clean',
                            'LibriSpeech/test-other',
                            'LibriSpeech/dev-clean',
                            'LibriSpeech/dev-other']


def extension_check(file, extension, file_use):
    """ Check argparse argument file extension. """
    if not file.endswith(extension):
        raise argparse.ArgumentTypeError('{} file must be of type {}'.format(file_use, extension))
    return file


def load_model(model_name):
    """ Load correct model based on model_name"""
    if model_name == 'quartznet':
        print('Using QuartzNet model')
        return stt_nemo.load_model(QUARTZNET_MODEL_PATH)
    elif model_name == 'jasper':
        print('Using Jasper model')
        return stt_nemo.load_model(JASPER_MODEL_PATH)
    elif model_name == 'deepspeech':
        print('Using DeepSpeech model')
        return stt_deepspeech.load_model(DEEPSPEECH_MODEL_PATH, DEEPSPEECH_SCORER_PATH)


def loop(model, model_name, input_path, output_path):
    """ Infinite loop that reads input audio file and transcribes it when changes are detected. """
    prev_mod_time = time.time()
    while True:
        # File location check
        if os.path.isfile(input_path):
            mod_time = os.stat(input_path)[8]
            if mod_time == prev_mod_time:
                print('Waiting for changes in input file', end='\r')
                time.sleep(1)
            else:
                prev_mod_time = mod_time
                print('\n Reading audio file')
                start_time = time.time()

                if model_name == 'deepspeech':
                    text = stt_deepspeech.speech_to_text(model, input_path)
                else:
                    text = stt_nemo.speech_to_text(model, input_path)

                print('Transcription finished in {} seconds'.format(np.round((time.time() - start_time), 3)))
                print('Output: {}'.format(text))

                # Write translation to output file
                with open(output_path, 'w') as file:
                    file.write(text)
        else:
            print('Input audio file not found, exiting')
            return


def write_results_to_file(path, results):
    with open(path, 'w') as file:
        file.write('dataset, WER (%), time per file\n')
        for i in range(len(results[0])):
            file.write(', '.join([str(results[0][i]), str(results[1][i]*100), ''.join([str(results[2][i]), '\n'])]))
    print('Evaluation results saved to', path)


def main():
    parser = argparse.ArgumentParser(description='Translate speech to text and save text to file')
    parser.add_argument('--i', metavar='INPUT', nargs='?', const=DEFAULT_INPUT_PATH,
                        help='Path to the input audio file (default: {})'.format(DEFAULT_INPUT_PATH),
                        type=lambda f: extension_check(f, '.wav', 'Input audio'))
    parser.add_argument('--o', metavar='OUTPUT', nargs='?', const=DEFAULT_OUTPUT_PATH,
                        help='Path to the output text file (default: {})'.format(DEFAULT_OUTPUT_PATH),
                        type=lambda f: extension_check(f, '.txt', 'Output'))
    parser.add_argument('-jasper', dest='model', action='store_const',
                        const='jasper', default='quartznet',
                        help='Use Jasper model (default: QuartzNet)')
    parser.add_argument('-deepspeech', dest='model', action='store_const',
                        const='deepspeech', default='quartznet',
                        help='Use DeepSpeech model (default: QuartzNet)')
    parser.add_argument('-evaluate', dest='evaluate', action='store_const',
                        const=True, default=False,
                        help='Evaluate model word error rate and time consumption. '
                             'Given INPUT and/or OUTPUT will be ignored')

    args = parser.parse_args()

    start_time = time.time()
    model = load_model(args.model)
    print('Model loaded in {} seconds'.format(np.round((time.time() - start_time), 3)))

    if args.evaluate:
        if args.model == 'deepspeech':
            write_results_to_file('/'.join(['LibriSpeech_results', '-'.join([args.model, 'evaluation.txt'])]),
                                  stt_deepspeech.evaluate(model, EVALUATION_DATASET_PATHS))
        else:
            write_results_to_file('/'.join(['LibriSpeech_results', '-'.join([args.model, 'evaluation.txt'])]),
                                  stt_nemo.evaluate(model, EVALUATION_DATASET_PATHS))
    else:
        loop(model, args.model, args.i, args.o)


if __name__ == '__main__':
    main()
