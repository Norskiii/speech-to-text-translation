import nemo.collections.asr as nemo_asr


def load_model(model_path):
    return nemo_asr.models.EncDecCTCModel.restore_from(model_path)


def speech_to_text(model, audio_file_path):
    return model.transcribe(paths2audio_files=[audio_file_path])[0]

