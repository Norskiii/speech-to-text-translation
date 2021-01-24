# Speech-to-text translation
Speech-to-text translation implimented as a part of a Deep Speaking Avatar Bsc project.

## DeepSpeech

Translation from speech to text is done using a open-source Speech-To-Text engine [DeepSpeech](https://github.com/mozilla/DeepSpeech). DeepSpeech has a pre-trained model and external scorer for English language, which can be downloaded with the following commands:
```
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```
DeepSpeech documentation encourages the use of virtual environments. More information on using a virtual environment and installing the DeepSpeech package can be found [here](https://deepspeech.readthedocs.io/en/v0.9.3/USING.html#using-the-python-package).

## Dependencies

Dependencies are defined in `requirements.txt` and `requirements_gpu.txt`. 

## Installation on Linux

1. Clone speech-to-text-translation to current directory:
```
$ git clone https://github.com/Norskiii/speech-to-text-translation.git
```
2. Create a virtual environment and activate it:

You can replace `$HOME/tmp/deepspeech-venv/` with any directory path.

```
$ virtualenv -p python3 $HOME/tmp/deepspeech-venv/
$ source $HOME/tmp/deepspeech-venv/bin/activate
```
3. Move to the project directory:
``` 
$ cd speech-to-text-translation 
```
4. Download the pre-trained model files:
```
$ wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
$ wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```
5. Install dependencies:

* Systems with the required [CUDA dependency](https://deepspeech.readthedocs.io/en/v0.9.3/USING.html#cuda-dependency-inference) and a supported NVIDIA GPU ([Compute Capability](https://developer.nvidia.com/cuda-gpus) at least 3.0):
``` 
$ pip3 install -r requirements_gpu.txt 
```
* Other systems:
```
$ pip3 install -r requirements.txt
```
