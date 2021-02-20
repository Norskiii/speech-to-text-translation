#------------------------------------------------------------
# Speech-to-text translation with pre-trained NeMo models
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
    # Model file locations
    quartznet_path = os.path.join(os.getcwd(),'QuartzNet15x5Base-En.nemo')
    jasper_path = os.path.join(os.getcwd(), 'Jasper10x5Dr-En.nemo')

    parser = argparse.ArgumentParser(description='Translate speech to text and save text to file')
    parser.add_argument('--i', metavar='INPUT', required=True,
                        help='Path to the input audio file (.wav)',
                        type=lambda f: extension_check(f, '.wav', 'Input audio'))
    parser.add_argument('--o', metavar='OUTPUT', required=True,
                        help='Path to the output text file (.txt)',
                        type=lambda f: extension_check(f, '.txt', 'Output'))
    parser.add_argument('-jasper', dest='model', action='store_const',
                        const='jasper', default='quartznet',
                        help='Use Jasper speech recognition model (default: QuartzNet)')

    args = parser.parse_args()
    
    # Load correct model
    if args.model == 'quartznet':
        print('Using QuartzNet model')
        model = nemo_asr.models.EncDecCTCModel.restore_from(quartznet_path)
    elif args.model == 'jasper':
        print('Using Jasper model')
        model = nemo_asr.models.EncDecCTCModel.restore_from(jasper_path)
    
    prev_modtime = time.time()
    while True:
        # File location check
        if os.path.isfile(args.i):
            modtime = os.stat(args.i)[8]
            if modtime == prev_modtime:
                print('Waiting for changes in input file', end='\r')
                time.sleep(1)
            else:
                prev_modtime = modtime
                print('\n Reading audio file')
                start_time = time.time()
                text = model.transcribe(paths2audio_files=[args.i])[0]
                print('Transcription finished in', str(np.round((time.time() - start_time), 3)), 'seconds')
                print(text)
                
                # Write translation to output file
                with open(args.o, 'w') as file:
                    file.write(text)
        else:
            print('Input audio file not found, exiting')
            return
        

if __name__ == '__main__':
    main()
