# Speech-to-text translation
Speech-to-text (STT) translation implimented as a part of a Deep Speaking Avatar Bsc project.

## DeepSpeech
In the baseline version, translation from speech to text is done using an open-source Speech-To-Text engine [DeepSpeech](https://github.com/mozilla/DeepSpeech). DeepSpeech has a pre-trained model and external scorer for English language, which can be downloaded with the following commands:
```
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```
## QuartzNet
QuartzNet is an automatic speech recognition model that is supported by [NVIDIA NeMo toolkit](https://github.com/NVIDIA/NeMo). It can achieve better word-error-rates when compared to DeepSpeech, so most likely it will be used in the final version of Deep Speaking Avatar STT.

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
   $ cd speech-to-text-translation 
   ```
3. Download the pre-trained DeepSpeech model files:
   ```
   $ wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
   $ wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
   ```
4. Install dependencies:

   * Systems with the required [CUDA dependency](https://deepspeech.readthedocs.io/en/v0.9.3/USING.html#cuda-dependency-inference) and a supported NVIDIA GPU    ([Compute Capability](https://developer.nvidia.com/cuda-gpus) at least 3.0):
      ``` 
      $ pip3 install -r requirements_gpu.txt 
      ```
   * Other systems:
      ```
      $ pip3 install -r requirements.txt
      ```
