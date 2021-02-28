# Speech-to-text translation
Speech-to-text (STT) translation implimented as a part of a Deep Speaking Avatar Bsc project.

## DeepSpeech
In the baseline version, translation from speech to text is done using an open-source Speech-To-Text engine [DeepSpeech](https://github.com/mozilla/DeepSpeech). 

## Jasper and QuartzNet
[NVIDIA NeMo toolkit](https://github.com/NVIDIA/NeMo) offers pre-build Jasper and QuartzNet speech recognition models. Both can achieve better word-error-rates when compared to DeepSpeech, so most likely one of them will be used for final model.

## Requirements
1. Python 3.6 or above
2. Pip3

## Setup on Linux
Run 'setup.sh' to install all needed dependencies and pre-trained models:
``` 
$ ./setup.sh
```

## How to use
``` 
usage: main.py [-h] [--i [INPUT]] [--o [OUTPUT]] [-jasper] [-deepspeech] [-evaluate]

Translate speech to text and save text to file

optional arguments:
  -h, --help    show this help message and exit
  --i [INPUT]   Path to the input audio file (default: /home/avatar/integration/stt_input.wav)
  --o [OUTPUT]  Path to the output text file (default: /home/avatar/integration/stt_output.txt)
  -jasper       Use Jasper model (default: QuartzNet)
  -deepspeech   Use DeepSpeech model (default: QuartzNet)
  -evaluate     Evaluate model word error rate and time consumption. Given INPUT and/or OUTPUT will be
                ignored
``` 
