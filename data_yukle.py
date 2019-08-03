# Mert Hacıahmetoğlu
# 03.08.2019

"""data loading script"""

#---------------------------------------------------------------------
"""GEREKLİ KÜTÜPHANELER"""
from __future__ import print_function
from hiperparametreler import Hiperparametreler as hp

import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata

def harf_yukle():
    char_idx = {char:idx for idx, char in enumerate(hp.harfler)} #karakter --> idx sözlük
    idx_char = {idx:char for idx, char in enumerate(hp.harfler)} #idx --> karakter sözlük
    return char_idx, idx_char

def metin_normalize(metin):
    text = ''.join(char for char in unicodedata.normalize('NFD', metin) 
                           if unicodedata.category(char) != 'Mn')

    metin=text.lower()
    metin = re.sub("[^{}]".format(hp.harfler), " ", metin)
    metin = re.sub("[ ]+", " ", metin)
    return metin

def yukle_data(mode="train"):
    """
    İki modla çalıştırılabilir, eğitim fonksiyonu
    mod:train ya da sentez.
    """

    char_idx, idx_char = harf_yukle()

    if mode=="train": #eğitim modu
        fpaths, metin_uzunlugu, metinler = [], [], []
        transkript = os.path.join(hp.data, 'transkript.csv')
        satirlar = codecs.open(transkript, 'r', 'utf-8').readlines()
        for satir in satirlar:
            fname, _, metin = line.strip().split("|")
            fpath = os.path.join(hp.data, "wavs", fname + ".wav")
            fpaths.append(fpath)

            metin = metin_normalize(metin) + "E"  # E: EndOfScript
            #her satırdan sonra cümlenin bittiği büyük e harfiyle belirtilmeli.
            metin = [char_idx[char] for char in metin] #idx'e çevirme işlemi
            metin_uzunlugu.append(len(metin))
            metinler.append(np.array(metin, np.int32).tostring())

        return fpaths, metin_uzunlugu, metinler

    else: # sentez modu
        satirlar = codecs.open(hp.test_data, 'r', 'utf-8').readlines()[1:] #ilk satırı atlayıp text dosyasını oku
        cumleler = [metin_normalize(line.split(" ", 1)[-1]).strip() + "E" for satir in satirlar] # text normalizasyon, E: EOS
        metinler = np.zeros((len(cumleler), hp.max_N), np.int32)
        for i, cumle in enumerate(cumleler):
            metinler[i, :len(cumle)] = [char_idx[char] for char in cumle] #metinler arrayinde cumle uzunlugu kadar veriyi
            # çıktıya uygun değiştir
        return metinler


def bacth_al():
    #datayı yığınlar (batch) halinde yükleme ve sıraya koyma fonksyonu


    with tf.device('/cpu:0'):
        fpaths, metin_uzunlugu, metinler = yukle_data() # liste
        maxlen, minlen = max(metin_uzunlugu), min(metin_uzunlugu)

        batch_s = len(fpaths) // hp.B #toplam alınacak batch sayısı

        fpath, metin_uzunlugu, metin = tf.train.slice_input_producer([fpaths, metin_uzunlugu, metinler], shuffle=True)

        metin = tf.decode_raw(metin, tf.int32)

        if hp.prepro: #preprocess açıksa;
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath)
                mel = "mels/{}".format(fname.decode("utf-8").replace("wav", "npy"))
                mag = "mags/{}".format(fname.decode("utf-8").replace("wav", "npy"))
                return fname, np.load(mel), np.load(mag)
            fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        else:
            print("!!!LUTFEN PREPRO AYARINI TRUE YAPIN!!!")

        fname.set_shape(())
        metin.set_shape((None,))
        mel.set_shape((None, hp.mels_s))
        mag.set_shape((None, hp.fft_s//2+1))

        _, (metinler, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=metin_uzunlugu,
                                            tensors=[metin, mel, mag, fname],
                                            batch_size=hp.B,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=8,
                                            capacity=hp.B*4,
                                            dynamic_pad=True)

    return metinler, mels, mags, fnames, num_batch