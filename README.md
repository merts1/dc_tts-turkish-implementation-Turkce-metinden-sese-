# dc_tts-turkish-implementation-Turkce-metinden-sese-
Guided attention metoduyla deep convolutional neural networks kullanılarak hazırlanan text to speech modelinin türkçeye derlenmiş halidir.
Aşağıda verilen konvolüsyonel ağları yardımıyla hazırlanmış araştırma yazısı kullanılarak oluşturulan metinden sese algoritmasının Kyubyong'a ait olan ingilizce kodunu türkçeye çevirip sesli kitap okuma videoları üzerine uygulayarak modelin türkçesini oluşturmaya çalıştım, modelin nasıl performans vereceğini merak ediyordum. Performans kısıtlamalarıma rağmen güzel sonuçlar elde ettim, düşük batch_size kullanıyor olmamdan dolayı guided attention metodunun vermesi gereken erken verimi alamadım. Araştırma kağıdında diğer RNN modellerine kıyasla çok daha kısa sürede kabul edilebilir sonuçlar aldığına değinilip 6gb vram bulunan çift gpu bilgisayarda 15 saat sonunda verinin eğitildiğini söylüyordular.
Ben 12 saatlik bir eğitim sonunda yaklaşık 500bin adım 1. aşama eğitimi yapıp ne kadar başarılı olduğunu kontrol ettim. Gerçekten de kabul edilebilir sonuçlar olduğu için 2. aşamada çözünürlük arttırmaya yarayan fazı da eğitip yayınlama kararı aldım. Ancak 2. aşamada batch size'ı daha da düşürmem gerekiyordu ve eğitim de gittikçe daha çok yükü gpu'ya bindiriyordu. O yüzden çözünürlük arttırma aşamasını yaklaşık 250bin adım kadar eğitebildim. Ancak yine de kabul edilebiilir sonuçlar verdi.
https://arxiv.org/pdf/1710.08969.pdf
https://github.com/Kyubyong
https://github.com/Kyubyong/dc_tts

Kodun ingilizce versiyonunda LJSpeech dataset kullanılmıştı, türkçede böyle bir kaynak bulamadığım için kolay anlaşıllır bir videodan altyazı yazmayı düşündüm ancak LJ datasetinde yaklaşık 13bin örnek olduğunu görünce bunun 4-5 ayımı alacağını farkettim ve hemen bir script yazdım.
Araştırma yazısında belirtildiği gibi 3-12 sn aralıklara 22050 sample rate ve 16bit wav audio file halinde split on silence işlemi uyguladım. Sonra bu ses dosyalarının her birini google voice sistemine yükleyip sesten metne dönüştüren kütüphaneleri kullandım. Böylece yaklaşık 2-3 saat içinde bilgisayar gerekli veriyi hazırlamıştı. Ancak elimde bulunan kod utf-8 decoding uyguladığı için kodun tamamını karıştırıp türkçe karakterleri eklemektense, o karakterlerin yerine sayı kullandım. Bu da sayı içeren cümlelerde bazı anlam bozukluklarına yol açtı, çünkü LJ datasetinde olduğu gibi sayıları ve kısaltmaları açma işlemi uygualamadım. Büyük harfleri düzeltmekle uğraşmadım çünkü kodda zaten buna yer verilmişti. Ayrıca modelin tanıdığı karakterlerden olmayan ama sık sık beliren ",'!() gibi işaretleri de kaldırmadım bu da karakter uzunluğunu etkilediği için modelin performansını etkilediğini düşünüyorum. Bu durumda daha çok veri toplayıp yüklesem ve bu eksiklikleri gidersem çok daha başarılı bir sonuç ortaya çıkabileceğini düşünüyorum.

Modelin 500bin adım eğitimden sonra ortaya koyduğu performansa aşağıdaki linkten bakabilirsiniz.
https://soundcloud.com/mert-hac-ahmeto-lu/sesler

Eğer siz de kendi modelinizi eğitmek istiyorsanız, sıfırdan başlamamanızı tavsiye ederim çünkü bu model ne kadar kötü olursa olsun en azından bazı şeylerin telaffuzunun öğrendi. Bu da eğitimi yaparken zaman kazanmanıza yardımcı olacaktır. Benimle aynı kişinin ses kaydını kullanmanıza gerek yok çünkü model konvolüsyonel ağlar kullandığından sonradan eğittiğiniz ses baskın olacaktır. Hatta yeterince iyi bir model eğittikten sonra 15bin 10bin gibi çok daha az adımlarla hernhangi birinin sesinin modelini eklerseniz o kişiye yakın sesler elde edebilirsiniz.

502bin adım eğitilmiş modelimin save dosyalarını aşağıdaki linke yükledim.


Başlıca gerekli kütüphaneler:
  1-librosa
  2-tqdm
  3-numpy
  4-matplotlib
  5-scipy
  6-tensorflow 1.3 ve üzeri

