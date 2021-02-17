#------------------------------------------------------------
# Speech-to-text translation with pre-trained Jasper model
#------------------------------------------------------------

import nemo
import nemo.collections.asr as nemo_asr
import os
import argparse
import time
import numpy as np


def extension_check(file, extension, file_use):
    """ Check argparse argument file extension """
    if not file.endswith(extension):
        raise argparse.ArgumentTypeError('{} file must be of type {}'.format(file_use, extension))
    return file


def main():
    # Model file location 
    model_path = os.path.join(os.getcwd(),'models/Jasper10x5Dr-En.nemo')

    # Load model
    jasper = nemo_asr.models.EncDecCTCModel.restore_from(model_path)
    
    parser = argparse.ArgumentParser(description='Translate speech to text and save text to file')
    parser.add_argument('--i', metavar='INPUT', required=True,
                        help='Path to the input audio file (.wav)',
                        type=lambda f: extension_check(f, '.wav', 'Input audio'))
    parser.add_argument('--o', metavar='OUTPUT', required=True,
                        help='Path to the output text file (.txt)',
                        type=lambda f: extension_check(f, '.txt', 'Output'))

    args = parser.parse_args()
    

    # file location check
    if os.path.isfile(args.i):
        print('Reading audio file')
        path = [os.path.join(os.getcwd(), args.i)]

        start_time = time.time()
        text = jasper.transcribe(paths2audio_files=path)[0]
        print('Transcription finished in', str(np.round((time.time() - start_time),3)), 'seconds')
    else:
        print('Input audio file not found, exiting')
        return
    
    # Write translation to output file
    with open(args.o, 'w') as file:
        file.write(text)
    
if __name__ == '__main__':
    main()
    
