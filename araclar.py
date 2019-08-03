# Mert Hacıahmetoğlu
# 03.08.2019

"""utilities script"""

#---------------------------------------------------------------------
from __future__ import print_function, division

import numpy as np
import librosa
import os, copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import signal

from hiperparametreler import Hiperparametreler as hp
import tensorflow as tf

def spektrogram_al(fpath):
	"""
	Data içerisinde bulunan wav dosyasını normalize edilmiş
	mel spektrograma ve lineer mutlak spektrograma dönüştür.

	Girdiler:
	fpath:ses dosyasının konumunu gösteren etiket

	Çıktılar:
	mel: 2boyutlu array (T, mels_s) float32.
	mag: 2boyutlu array (T, 1+fft_s/2) float32.
	"""

	y, sr = librosa.load(fpath, sr=hp.sr)
	#ses dosyasını yükle

	y, _ = librosa.effects.trim(y)
	#trim işlemi

	y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])
	#preemphasis (önvurgu) / Sinyal gürültü oranı arttırma

	linear = librosa.stft(y=y, fft_s=hp.fft_s,
                          shift=hp.shift,
                          win_length=hp.win_length)
	#stft : Kısa zamanlı Fourier dönüşümü

	"""AçıklamaKısa zamanlı Fourier dönüşümü, zamana bağlı değişen 
	bir sinüzoidal sinyalin yerel bölümlerinin frekansını ve fazını 
	bulmakta kullanılan bir tür Fourier dönüşümüdür"""

	mag = np.abs(linear)  # (1+fft_s//2, T)
	#mutlak spektrogramı (mag)

	mel_temel = librosa.filters.mel(hp.sr, hp.fft_s, hp.mels_s)  # (mels_s, 1+fft_s//2)
	mel = np.dot(mel_temel, mag)  # (mels_s, t)
	#mel spektrogramı

	mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))
    #desibel dönüşümü

    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    #normalize işlemi

    mel = mel.T.astype(np.float32)  # (T, mels_s)
    mag = mag.T.astype(np.float32)  # (T, 1+fft_s//2)
    #transpozunu alma işlemi ve astype float32

    return mel, mag

def spektrogram2ses(mag):
	"""
	Lineer mutlak spektrogramı kullanarak wav ses dosyası oluşturmaya
	yarayan fonksiyon.

	Girdiler:
	mag: 2boyutlu array (T, 1+fft_s/2)

	Çıktılar:
	wav: 1boyutlu numpy array.
	"""

	mag=mag.T
	mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db
	# normalize-denormalize

	mag = np.power(10.0, mag * 0.05)
	# genlik

	wav = griffin_lim(mag**hp.power)
	# wav ses dosyası için time-series elde etme algoritması, aşağıda define edilmiştir.

	wav = signal.lfilter([1], [1, -hp.preemphasis], wav)
	# geri önvurgu

	wav, _ = librosa.effects.trim(wav)

	return wav.astype(np.float32)

def griffin_lim(spektrogram):
	"""Griffin-Lim's algorithm."""

	X_best = copy.deepcopy(spektrogram)
	for i in range(hp.iter_s):
		X_t = invert_spectrogram(X_best)
		tahmin = librosa.stft(X_t, hp.fft_s, hp.shift, win_length=hp.win_length)
		faz = tahmin / np.maximum(1e-8, np.abs(tahmin))
		X_best = spektrogram * faz
	X_t = invert_spectrogram(X_best)
	y = np.real(X_t)
	return y

def tersal_spektrogram(spektrogram):
	#inverse fft uygulaması:

	return librosa.istft(spektrogram, hp.shift, win_length=hp.win_length, window="hann")
	#stft window function tablo 1de hanning olarak belirlenmiştir.

def plot_olustur(alignment, gs, dir=hp.kayitlar):
	"""
	spektrogramın plot'ını çizen fonksiyon.

	Girdiler:
	alignment: encoder ve decoder adımlarını içeren numpy array
	gs:global step
	dir: kayıtların tutulacağı klasör

	"""

	if not os.path.exists(dir): os.mkdir(dir) #klasör oluştur
	
	fig, ax = plt.subplots()
	im = ax.imshow(alignment)
	fig.colorbar(im)
	plt.title('{} Adımlar'.format(gs))
	plt.savefig('{}/Adım-{}.png'.format(dir, gs), format='png')
	plt.close(fig)

def guided_attention(g=0.2): #g is set to 0.2 in the paper
	"""Guided attention bknz:sayfa 3"""
	W = np.zeros((hp.max_N, hp.max_T), dtype=np.float32)
	for n_pos in range(W.shape[0]):
		for t_pos in range(W.shape[1]):
			W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(hp.max_T) - n_pos / float(hp.max_N)) ** 2 / (2 * g * g))
			#sayfa3,sağdan ikinci paragraf formül: Wnt = 1 − exp{[−(n/N − t/T)^2] / 2*(g)^2}
	return W

def learning_rate_decay(init_lr, global_step, warmup_steps = 4000.0):
	step = tf.to_float(global_step + 1)
	return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
	#learning rate decay fonksiyonu

def spektrogram_yukle(fpath):
	fname = os.path.basename(fpath)
	mel, mag = spektrogram_al(fpath)
	t = mel.shape[0]

	#reduction ayarı için Marginal padding:
	padding_sayısı = hp.r - (t % hp.r) if t % hp.r != 0 else 0
	mel = np.pad(mel, [[0, padding_sayısı], [0, 0]], mode="constant")
	mag = np.pad(mag, [[0, padding_sayısı], [0, 0]], mode="constant")

	#reduction:
	mel = mel[::hp.r, :]
	return fname, mel, mag