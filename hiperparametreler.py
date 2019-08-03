# Mert Hacıahmetoğlu
# 03.08.2019

"""hyperparameters script"""

#---------------------------------------------------------------------

class Hiperparametreler:
    prepro = True # training öncesi datayı düzgünce işlenebilecek 
    #             spektrogramlara çevirmek için yapılan preprocess.

    #------------------------------------------------------------------

    #sinyal ayrıştırma: bu parametreler araştırma kağıdı üzerindeki tablo 1
    #baz alınarak düzenlenmiştir.
    sr = 22050 # eslere ait sampling rate
    fft_s = 2048 # fast fourier transform nokta sayısı (hızlı fourier dönüşümü)
    frame_shift = 0.0125 # tablo 1de yer alan 11.6ms değerinin 12.5a yuvarlanmış saniye gösterimi
    frame_length = 0.05 # yine aynı tabloda 46.4ms değerinin 50ye yuvarlanmış saniye gösterimi

    win_length = int(frame_length * sr)
    shift = int(frame_shift * sr)
    # yeniden yuvarlanmış değerlere göre yeni win length and shift değerleri

    mels_s = 80 # mel spektrogram boyutu (80 * T, T süre olmak üzere)
    power = 1.5 # 
    iter_s = 50 # griffin lim sırasında yapılması gereken inversiyon adedi
    #griffin lim araclar scriptinde tanımlandı.
    preemphasis = .97 #özvurgu değeri

    max_db = 100
    ref_db = 20
    #bu iki değer ses dosyalarını eğitime sokmadan önce normalize etmeye yarayacak olan
    #minimum ve maksimum desibel değerleridir.

    #-------------------------------------------------------------------------

    r = 4 #reduction factor, eğitimi hızlandırmak için dimension azaltıcı faktör.
    dropout_rate= .05
    e = 128 #tablo-1
    d = 256 #tablo-1 hidden units, Text2Mel
    c = 512 #tablo-1 hidden units, SSRN
    attention_win_size = 3

    #-------------------------------------------------------------------------

    # data:
    data = "Sesler-tr"
    test_data = "ornek_cumleler.txt"
    harfler = "PE abc0defg1h2ijklmno3pqrs4tu5vwxyz'.?" # P: Padding, E: EOS.
    # burada kodu utf-8 şeklinde kodlayıp muhtemel bir hatayla uğraşmamak için
    # türkçe karakterlerin bazıları yerine sayı kullandım
    # { 0:ç , 1:ğ , 2:ı , 3:ö , 4:ş , 5:ü }

    max_N = 180 # attention grafiğinde istenen maximum karakter sayısı
    max_T = 210 # max mel frame adedi

    #--------------------------------------------------------------------------

    #training:
    LEARNING_RATE = .001
    kayitlar = "kayitlar/sesler-01"
    ornekler = "ornekler"
    B = 16 # batch size, önerilen eğitimin 2. aşamasında 8 yapılabilir
    num_iterations = 2000000