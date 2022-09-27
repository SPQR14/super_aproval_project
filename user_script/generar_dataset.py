import librosa
import os
import numpy as np
import pandas as pd
import time
import librosa.display
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

start = time.process_time()

os.chdir(os.getcwd())
data_path = '../data/audio/'
plot_path = '../data/spectrograms/'

print('Directorio actual: ', os.getcwd())

tag = ['mg', 'no']
hop_length = 512
n0 = 9000
n1 = 9100

columnas = ['archivo' ,'zero_cr' ,'spectral_centroid', 'spectral_bw' ,'spectral_rf', 'croma', 'norm_main_signal','norm_amplitude_db']
for x in range(1, 21):
    columnas.append(f'mfcc_{x}')
columnas.append('BPM')
columnas.append('auto_c')
columnas.append('target')

df = pd.DataFrame(columns = columnas)
for g in tag:
    for nombre in os.listdir(f'{data_path}{g}'):
        cancion = f"{data_path}{g}/{nombre}"
        print('Pista actual', cancion)
        samples, sr = librosa.load(cancion, sr = None, mono = True, offset = 0.0, duration = None)
        nombre = nombre.replace(' ', '')
        zero_crossings = librosa.zero_crossings(samples[n0:n1], pad=False)
        zero_crossings = sum(zero_crossings)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y = samples, sr = sr))
        spectral_bw = np.mean(librosa.feature.spectral_bandwidth(y = samples, sr = sr))
        spectral_rf = np.mean(librosa.feature.spectral_rolloff(y = samples, sr = sr))
        croma = np.mean(librosa.feature.chroma_stft(y = samples, sr = sr))
        mfcc = librosa.feature.mfcc(y = samples, sr = sr)
        env = librosa.onset.onset_strength(y = samples, sr = sr, hop_length = hop_length)
        tempograma = librosa.feature.tempogram(onset_envelope = env, sr = sr, hop_length = hop_length)
        auto_c = librosa.autocorrelate(env, max_size = tempograma.shape[0])
        auto_c = librosa.util.normalize(auto_c)
        auto_c = np.mean(auto_c)
        BPM = librosa.beat.tempo(onset_envelope = env, sr = sr, hop_length = hop_length)[0]
        
        main_signal = librosa.stft(samples)
        amplitude = np.linalg.norm(librosa.amplitude_to_db(abs(main_signal)))

        x = f'{nombre} {zero_crossings} {spectral_centroid} {spectral_bw} {spectral_rf} {croma} {np.linalg.norm(main_signal)} {amplitude}'

        for m in mfcc:
            x += f' {np.mean(m)}'
        
        x += f' {BPM} {auto_c} {g} '

        d = dict(zip(columnas, x.split()))
        df = df.append(d, ignore_index = True)
        print(u'Añadido el registro de: ' + nombre)
        
df.to_csv('datos_musica_new.csv', sep = ',', encoding = 'utf8')
print(time.process_time() - start)
