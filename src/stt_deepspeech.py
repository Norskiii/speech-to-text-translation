import numpy as np
import librosa as lb
from deepspeech import Model
import time
import os
import jiwer


def load_model(model_path, scorer_path):
    model = Model(model_path)
    model.enableExternalScorer(scorer_path)
    return model


def speech_to_text(model, audio_file_path):
    audio, audio_sample_rate = lb.load(audio_file_path, sr=None)

    if audio_sample_rate != model.sampleRate():
        print('Warning: input audio sample rate ({}) is different than the desired sampling rate ({}). '
              'Resampling might affect speech recognition.'.format(audio_sample_rate, model.sampleRate()))
        audio = lb.resample(audio, audio_sample_rate, model.sampleRate())

    # DeepSpeech requires the audio to be type int16
    audio = (audio * 32768).astype(np.int16)

    return model.stt(audio)


def evaluate(model, dataset_paths):
    """ Evaluate model with given datasets. """
    datasets = []
    wer = []
    time_per_file = []

    for j in range(len(dataset_paths)):
        datasets.append(dataset_paths[j].rsplit('/', 1)[1])

        # Used when calculating time_per_file for each dataset
        time_per_file.append(0)
        files_total = 0

        s = 0       # num of substitutions
        d = 0       # num of deletions
        i = 0       # num of insertions
        c = 0       # num of correct words

        for folder in os.listdir(dataset_paths[j]):
            folder_path = os.path.join(dataset_paths[j], folder)
            for sub_folder in os.listdir(folder_path):
                sub_folder_path = os.path.join(folder_path, sub_folder)

                # .trans.txt contains the names and ground truths for audio files in sub_folder
                trans_path = os.path.join(sub_folder_path, str(folder + '-' + sub_folder + '.trans.txt'))

                with open(trans_path) as file:
                    lines = file.readlines()

                audio_files, ground_truths = zip(*([x.split(' ', 1) for x in lines]))
                audio_file_paths = [os.path.join(sub_folder_path, x+'.flac') for x in audio_files]

                # Pre-process ground truth values
                ground_truths = [x.lower().strip() for x in ground_truths]

                print('Transcribing audio files found under', sub_folder_path)
                for x in range(len(audio_file_paths)):
                    print('Working ...')
                    start_time = time.time()
                    transcription = speech_to_text(model, audio_file_paths[x]).lower().strip()
                    time_per_file[j] += (time.time() - start_time)
                    files_total += 1

                    # Computing wer from all dataset audio transcriptions causes a MemoryError so it is done in parts
                    measures = jiwer.compute_measures(ground_truths[x], transcription)

                    s += measures['substitutions']
                    d += measures['deletions']
                    i += measures['insertions']
                    c += measures['hits']

        time_per_file[j] = time_per_file[j]/files_total
        wer.append(float(s + d + i) / float(s + d + c))

    return [datasets, wer, time_per_file]
