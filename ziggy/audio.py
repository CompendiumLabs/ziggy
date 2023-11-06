# audio tools

import re
import time
import torch
import sounddevice as sd

from tqdm import tqdm
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

##
## constants
##

WHISPER_RATE = 16000

##
## utilities
##

def timer_bar(seconds, tick=0.01, jiffy=0.001, **kwargs):
    args = {
        'bar_format': '0 |{bar}| {total:.0f}', 'leave': False,
        'unit_scale': tick, 'ncols': 80, 'mininterval': tick, **kwargs
    }
    targ = time.time()
    for i in tqdm(range(int(seconds/tick)), **args):
        targ += tick
        while time.time() < targ:
            time.sleep(jiffy)

##
## speech-to-text model
##

class WhisperModel:
    def __init__(self, model='large', device='cuda'):
        import whisper

        # load pretrained model
        self.model = whisper.load_model(model, device=device)

    def transcribe(self, source=None, duration=None):
        if source is None:
            # get the number of ticks to record
            ticks = int(duration * WHISPER_RATE)

            # record the audio and show progress
            wave = sd.rec(ticks, samplerate=WHISPER_RATE, channels=1)
            timer_bar(duration)
            sd.wait()

            # pass to whisper for transcription
            form = torch.tensor(wave[:,0], device=self.model.device)
            result = self.model.transcribe(form)
        else:
            result = self.model.transcribe(source)

        # return text
        return result['text'].lstrip()

##
## text-to-speech model
##

class SpeechT5Model:
    def __init__(self, device='cuda'):
        self.device = device
        self.processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
        self.model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts').to(device)
        self.vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan').to(device)
        speakers = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')
        self.xvector = torch.tensor(speakers['xvector'], device=device)

    def synthesize(self, text, voice=7307, blocking=False, play=True, split=True):
        # break into sentences and process
        if split:
            texts = re.split(r' *[\.\n]+ *', text)
            wave = torch.cat([
                self.synthesize(t, voice=voice, play=False, split=False) for t in texts
            ])
        else:
            tokens = self.processor(text=text, return_tensors='pt')['input_ids'].to(self.device)
            wave = self.model.generate_speech(tokens, self.xvector[[voice],:], vocoder=self.vocoder)

        # play or return waveform
        if play:
            sd.play(wave.cpu().numpy(), samplerate=WHISPER_RATE, blocking=blocking)
        else:
            return wave
