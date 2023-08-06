# audio tools

import time
import torch
import whisper
import sounddevice as sd
from tqdm import tqdm

##
## constants
##

WHISPER_RATE = 16000

##
## utilities
##

def timer_bar(seconds, tick=0.1, **kwargs):
    args = {
        'bar_format': '0 |{bar}| {total:.0f}', 'leave': False,
        'unit_scale': tick, 'ncols': 80, 'mininterval': tick, **kwargs
    }
    for i in tqdm(range(int(seconds/tick)), **args):
        time.sleep(tick)

##
## speech-to-text model
##

class WhisperModel:
    def __init__(self, model='large', device='cuda'):
        self.model = whisper.load_model(model, device=device)

    def transcribe(self, duration):
        # get the number of ticks to record
        ticks = int(duration * WHISPER_RATE)

        # record the audio and show progress
        wave = sd.rec(ticks, samplerate=WHISPER_RATE, channels=1)
        timer_bar(duration)
        sd.wait()

        # pass to whisper for transcription
        form = torch.tensor(wave[:,0], device=self.model.device)
        result = self.model.transcribe(form)

        # return text
        return result['text'].lstrip()
