import nemo.collections.asr as nemo_asr
import os
import time
import jiwer


def load_model(model_path):
    """ Load ASR model from model_path. """
    return nemo_asr.models.EncDecCTCModel.restore_from(model_path)


def speech_to_text(model, audio_file_path):
    """ Perform speech-to-text translation for one audio file. """
    return model.transcribe(paths2audio_files=[audio_file_path])[0]


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

                print('Transcribing audio files found under', sub_folder_path)
                start_time = time.time()
                transcriptions = model.transcribe(paths2audio_files=audio_file_paths)
                time_per_file[j] += (time.time() - start_time)
                files_total += len(transcriptions)

                ground_truths = [x.lower().strip() for x in ground_truths]
                transcriptions = [x.lower().strip() for x in transcriptions]

                # Computing wer from all dataset audio transcriptions causes a MemoryError so it is done in parts
                measures = jiwer.compute_measures(list(ground_truths), transcriptions)

                s += measures['substitutions']
                d += measures['deletions']
                i += measures['insertions']
                c += measures['hits']

        time_per_file[j] = time_per_file[j]/files_total
        wer.append(float(s + d + i) / float(s + d + c))

    return [datasets, wer, time_per_file]
