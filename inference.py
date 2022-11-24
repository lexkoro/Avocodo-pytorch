from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import glob
import json
import os

import librosa
import torch
from env import AttrDict
from meldataset import MAX_WAV_VALUE, load_wav, mel_spectrogram
from scipy.io.wavfile import write
from torchaudio import transforms

from models import Generator


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x, h):
    return mel_spectrogram(
        x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax
    )


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "*")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ""
    return sorted(cp_list)[-1]


checkpoint_file = "/home/akorolev/master/projects/vits-emotts/vocoder/g_00860000"
config_file = os.path.join(os.path.split(checkpoint_file)[0], "config.json")
with open(config_file) as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)

device = torch.device("cpu")
generator = Generator(h).to(device)
state_dict_g = load_checkpoint(checkpoint_file, device)
generator.load_state_dict(state_dict_g["generator"])
resampler = transforms.Resample(16000, 44100)

generator.eval()
generator.remove_weight_norm()
with torch.no_grad():
    filename = "/home/akorolev/master/projects/data/SpeechData/tts/EmotionData/EMO-DB/15a04Ac.wav"
    wav, sr = librosa.load(filename, sr=16000)
    wav = torch.FloatTensor(wav).to(device)
    wav = resampler(wav).to(device)

    x = get_mel(wav.unsqueeze(0), h)
    y_g_hat, _, _ = generator(x)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype("int16")

    output_file = os.path.join(
        "/home/akorolev/master/projects/vits-emotts/vocoder/output_files",
        os.path.splitext(filename)[0] + "_generated.wav",
    )
    write(output_file, h.sampling_rate, audio)
