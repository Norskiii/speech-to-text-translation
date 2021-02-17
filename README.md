# Speech-to-text translation
Speech-to-text (STT) translation implimented as a part of a Deep Speaking Avatar Bsc project.

## DeepSpeech
In the baseline version, translation from speech to text is done using an open-source Speech-To-Text engine [DeepSpeech](https://github.com/mozilla/DeepSpeech). DeepSpeech has a pre-trained model and external scorer for English language, which can be downloaded with the following commands:
```
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```
## Jasper and QuartzNet
[NVIDIA NeMo toolkit](https://github.com/NVIDIA/NeMo) offers pre-build Jasper and QuartzNet speech recognition models. Both can achieve better word-error-rates when compared to DeepSpeech, so most likely one of them will be used for final model.

## Requirements
1. Python 3.6 or above
2. Pytorch 1.7.1 or above

## Setup on Linux

1. Clone speech-to-text-translation to current directory:
   ```
   $ git clone https://github.com/Norskiii/speech-to-text-translation.git
   ```
2. Move to the project directory:
   ``` 
   $ cd speech-to-text-translation/models
   ```
3. Download the pre-trained models:
   * DeepSpeech model:
      ```
      $ wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
      $ wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
      ```
   * Jasper model: 
      ```
      $ wget https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/Jasper10x5Dr-En.nemo
      ```
   * QuartzNet model:
      ```
      $ wget https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-En.nemo
      ```
4. Install dependencies:
   ```
   $ pip3 install -r requirements.txt
   ```
