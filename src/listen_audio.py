#-------------------------------------------------
# Uses user input to listen and save 5s audio clip
#-------------------------------------------------

import sounddevice as sd
import soundfile as sf

samplerate = 16000  # Hertz
duration = 5  # seconds
filename = '/home/avatar/integration/stt_input.wav'

while(True):
    choice = input("Press 'r' to record 5 second audio clip or 'q' to exit \n")
    if choice == 'r':
        mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
                channels=1, blocking=True)
        choice = input("Recording ended, press 's' to save \n")
        if choice == 's':
            sf.write(filename, mydata, samplerate)

    elif choice == 'q':
        break



