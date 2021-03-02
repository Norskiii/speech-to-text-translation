import os
import soundfile as sf
import numpy as np
from tabulate import tabulate


def get_audio_durations(audio_paths):
    """ Return audio file durations in seconds for all files in audio_paths. """
    durations = []
    for path in audio_paths:
        audio, samplerate = sf.read(path)
        durations.append(len(audio)/float(samplerate))

    return durations


def get_audio_lengths(transcriptions):
    """ Return audio file transcription lengths in words for all sentences in transcriptions. """
    lengths = []
    for transcription in transcriptions:
        words = transcription.split()
        lengths.append(len(words))

    return lengths


def print_duration_information(dataset_names, num_of_files, mean_d, min_d, max_d, all_audio_d):
    headers = ['Dataset', 'num of audio files', 'mean duration (s)', 'min duration (s)', 'max duration (s)']
    table = []

    for i in range(len(dataset_names)):
        table.append([dataset_names[i], num_of_files[i], mean_d[i], min_d[i], max_d[i]])

    table.append(['all datasets', len(all_audio_d), np.mean(all_audio_d), np.min(all_audio_d), np.max(all_audio_d)])
    print(tabulate(table, headers=headers))


def print_length_information(dataset_names, num_of_files, mean_l, min_l, max_l, all_audio_l):
    headers = ['Dataset', 'num of transcriptions', 'mean length (words)', 'min length (words)', 'max length (words)']
    table = []

    for i in range(len(dataset_names)):
        table.append([dataset_names[i], num_of_files[i], mean_l[i], min_l[i], max_l[i]])

    table.append(['all datasets', len(all_audio_l), np.mean(all_audio_l), np.min(all_audio_l), np.max(all_audio_l)])
    print(tabulate(table, headers=headers))


def main():
    """ Displays statistics of the LibriSpeech datasets used in evaluating STT models. """
    librispeech_path = os.path.join(os.getcwd(), '..', 'LibriSpeech')

    dataset_names = []
    num_of_files = []

    # Mean, min and max audio file durations in seconds for all datasets
    mean_d = []
    min_d = []
    max_d = []

    # Mean, min and max audio file transcription lengths in words for all datasets
    mean_l = []
    min_l = []
    max_l = []

    # All audio file durations in seconds for all datasets
    all_audio_d = []

    # All audio file transcription lengths in words for all dataset
    all_audio_l = []

    for dataset in os.listdir(librispeech_path):
        if dataset.endswith('.TXT'):
            continue
        print('Reading audio files under', dataset)
        audio_d = []
        audio_l = []
        dataset_names.append(dataset)
        dataset_path = os.path.join(librispeech_path, dataset)
        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            for sub_folder in os.listdir(folder_path):
                sub_folder_path = os.path.join(folder_path, sub_folder)
                # .trans.txt contains the names and ground truths for audio files in sub_folder
                trans_path = os.path.join(sub_folder_path, str(folder + '-' + sub_folder + '.trans.txt'))

                with open(trans_path) as file:
                    lines = file.readlines()

                audio_files, ground_truths = zip(*([x.split(' ', 1) for x in lines]))
                audio_file_paths = [os.path.join(sub_folder_path, x + '.flac') for x in audio_files]

                audio_d.extend(get_audio_durations(audio_file_paths))
                audio_l.extend((get_audio_lengths(ground_truths)))

        num_of_files.append(len(audio_d))

        mean_d.append(np.mean(audio_d))
        min_d.append(np.min(audio_d))
        max_d.append(np.max(audio_d))
        all_audio_d.extend(audio_d)

        mean_l.append(np.mean(audio_l))
        min_l.append(np.min(audio_l))
        max_l.append(np.max(audio_l))
        all_audio_l.extend(audio_l)

    print(' ')
    print(' ')
    print_duration_information(dataset_names, num_of_files, mean_d, min_d, max_d, all_audio_d)
    print(' ')
    print(' ')
    print_length_information(dataset_names, num_of_files, mean_l, min_l, max_l, all_audio_l)


if __name__ == '__main__':
    main()
