import librosa
import os
import time
import librosa.display
import matplotlib.pyplot as plt
import pathlib

import warnings
warnings.filterwarnings("ignore")

start = time.process_time()

os.chdir(os.getcwd())
data_path = '../data/audio/'
plot_path = '../data/spectrograms/'

print('Directorio actual: ', os.getcwd())

tag = ['mg', 'no']
plt.figure(figsize=(20, 8))

for g in tag:
    pathlib.Path('../epectrogramas/' + g).mkdir(parents = True, exist_ok = True)
    for nombre in os.listdir(f'{data_path}{g}'):
        cancion = f"{data_path}{g}/{nombre}"
        print('Pista actual', cancion)
        samples, sr = librosa.load(cancion, sr = None, mono = True, offset = 0.0, duration = None)
        X = librosa.stft(samples)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis ='log')
        plt.title(nombre)
        plt.savefig(plot_path + g + '/' + nombre + '.png')
        plt.clf()
        

print(time.process_time() - start)
