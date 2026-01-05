# This is a sample Python script.
# https://lumo.proton.me/u/4/c/7e392470-e4cc-474f-a4e2-32344ae332a8

import librosa
import numpy as np
import soundfile as sf

# -------------------------------------------------
# 1. Caricamento del file audio
# -------------------------------------------------
# y = segnale mono, sr = frequenza di campionamento
y, sr = librosa.load('sample/sax.wav', sr=None, mono=True)   # sr=None mantiene la SR originale

# -------------------------------------------------
# 2. DFT (FFT) del segnale
# -------------------------------------------------
Y = np.fft.rfft(y)                     # rfft restituisce solo la metà positiva (efficiente)
freqs = np.fft.rfftfreq(len(y), d=1/sr)   # vettore delle frequenze corrispondenti

# -------------------------------------------------
# 3. Filtro passa‑basso (esempio: taglia > 4 kHz)
# -------------------------------------------------
cutoff = 4000.0                         # Hz
mask = freqs <= cutoff                  # True per le frequenze da mantenere
Y_filtered = Y * mask                   # azzera le componenti oltre il cutoff

# -------------------------------------------------
# 4. Trasformata inversa per ottenere il segnale filtrato
# -------------------------------------------------
y_filtered = np.fft.irfft(Y_filtered, n=len(y))

# Normalizzazione opzionale (evita clipping)
y_filtered = y_filtered / np.max(np.abs(y_filtered))

# -------------------------------------------------
# 5. Salvataggio del risultato
# -------------------------------------------------
sf.write('out/sax.wav', y_filtered, sr)
print('Filtro applicato e file salvato come my_track_filtered.wav')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
