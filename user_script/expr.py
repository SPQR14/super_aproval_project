import librosa
import os
import numpy as np
import pandas as pd
import eyed3
import time

import warnings
warnings.filterwarnings("ignore")

start = time.process_time()

os.chdir(os.getcwd())
data_path = '../data/audio/mg'

print('Directorio actual: ', os.getcwd())

tag = ['mg', 'no']
hop_length = 512
n0 = 9000
n1 = 9100

columnas = ['archivo' ,'zero_cr' ,'spectral_centroid', 'spectral_bw' ,'spectral_rf', 'croma']
for x in range(1, 21):
    columnas.append(f'mfcc_{x}')
columnas.append('BPM')
columnas.append('auto_c')
columnas.append('genero')


pistas = os.listdir(data_path)

print('Pista actual', pistas[24])
samples, sr = librosa.load(pistas[24], sr = None, mono = True, offset = 0.0, duration = None)
X = librosa.stft(samples)
Xdb = librosa.amplitude_to_db(abs(X))

print(np.linalg.norm(Xdb))
print('**************')
print(len(X), len(Xdb))