import numpy as np
import librosa as lb
from deepspeech import Model


def load_model(model_path, scorer_path):
    model = Model(model_path)
    model.enableExternalScorer(scorer_path)
    return model


def speech_to_text(model, audio_file_path):
    audio, audio_sample_rate = lb.load(audio_file_path, sr=None)

    if audio_sample_rate != model.sampleRate():
        print('Warning: input audio sample rate ({}) is different than the desired sampling rate ({}). '
              'Resampling might affect speech recognition.'.format(audio_sample_rate, model.sampleRate()))
        audio = lb.resample(audio, audio_sample_rate, model.sampleRate)

    # DeepSpeech requires the audio to be type int16
    audio = (audio * 32768).astype(np.int16)

    return model.stt(audio)
