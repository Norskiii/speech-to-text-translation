# Speech-to-text translation
Speech-to-text translation implimented as a part of a Deep Speaking Avatar Bsc project.

## DeepSpeech

Translation from speech to text is done using a open-source Speech-To-Text engine [DeepSpeech](https://github.com/mozilla/DeepSpeech). DeepSpeech has a pre-trained model for English language, which can be downloaded using the following commands:
```
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```
For the code to work both the model and DeepSpeech python package need to be downloaded and/or installed. DeepSpeech documentation also encourages the use of virtual environments. More information on using a virtual environment and installing the package can be found [here](https://deepspeech.readthedocs.io/en/v0.9.3/USING.html#using-the-python-package).
