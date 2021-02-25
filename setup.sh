# Install packages required by Nemo Toolkit
sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg

# Install pip requirements
pip3 install -r requirements.txt

# Get pre-trained models
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
wget https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/Jasper10x5Dr-En.nemo
wget https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-En.nemo

