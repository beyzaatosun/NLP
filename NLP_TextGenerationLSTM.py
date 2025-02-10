import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

texts = [
    "Bugün hava çok güzel dışarıda yürüyüş yapmayı düşünüyorum",
    "Kitap okumak beni gerçekten mutlu ediyor",
    "Yeni bir film izledim, çok beğendim",
    "Arkadaşlarımla buluşmayı sabırsızlıkla bekliyorum",
    "Yaz tatilini çok seviyorum, denize girmeyi dört gözle bekliyorum",
    "Geçen hafta yeni bir yemek tarifi denedim, harika oldu",
    "Spor salonuna gitmek, günün stresini atmak için çok iyi bir yöntem",
    "Yaz akşamlarında açık havada film izlemek çok keyifli",
    "Şu an bir proje üzerinde çalışıyorum ve sonuçları merakla bekliyorum",
    "Kendi başıma keşfettiğim yeni bir müzik türü var, çok hoşuma gitti",
    "Bugün yeni bir restoran keşfettim, yemekler gerçekten çok lezzetliydi",
    "Hava çok soğuk, içeri girip sıcak çikolata içmeyi düşünüyorum",
    "Arkadaşlarımla gezmeye gitmek için hafta sonunu sabırsızlıkla bekliyorum",
    "Yeni bir dil öğrenmeye başladım, ilerlemek çok heyecan verici",
    "Kısa bir tatil planlıyorum, birkaç günlüğüne deniz kenarına gideceğim",
    "Bugün sabah erken saatlerde kahvemi içip bir süre sessiz kaldım",
    "Farklı kültürleri tanımak, gezmek çok keyifli bir deneyim",
    "Birkaç yeni hobim oldu, müzikle ilgilenmek bunlardan biri",
    "Dışarıda çok güzel bir gün, parkta yürüyüş yapmayı planlıyorum",
    "Geçen gün eski arkadaşımı gördüm, uzun zaman sonra buluştuk",
    "Sabahları güne başlamak için spor yapmak bana iyi geliyor",
    "Kendime bir gün ayırıp kitap okumayı düşünüyorum",
    "Yazın geç vakitlerde sahilde yürüyüş yapmak harika bir şey",
    "Bugün yeni bir konu hakkında araştırma yapmaya başladım",
    "Yaz tatilinde denize girmek çok keyifli, sabırsızlanıyorum",
    "Hafta sonu için sinemaya gitmeyi düşünüyorum",
    "Bugün hava çok soğuk ama yine de dışarıda zaman geçirmeyi seviyorum",
    "Akşam yemeği için yeni bir tarif denemeyi planlıyorum",
    "Birkaç gündür sağlıklı yemekler yapmaya özen gösteriyorum",
    "Bu akşam arkadaşlarımla akşam yemeği yiyeceğiz, çok heyecanlıyım",
    "Sabahları erken uyanıp güne başlamak bana motivasyon sağlıyor",
    "Bir kitapçıda gezip yeni kitaplar almak gerçekten çok keyifli",
    "Kısa bir tatil planlıyorum, doğal alanlarda zaman geçirmek istiyorum",
    "Bugün havalar çok güzel, piknik yapmayı düşünüyorum",
    "Evde yoga yapmayı denedim, çok rahatlatıcıydı",
    "Sabah kahvemi içerken birkaç sayfa kitap okurum, çok güzel bir alışkanlık",
    "Yeni bir müzik grubu keşfettim, şarkılarını dinlemek çok eğlenceli",
    "Dışarıda yürüyüş yapmak, günün stresini atmak için harika bir yöntem",
    "Geceyi açık havada geçirip yıldızları izlemek çok huzur verici",
    "Bugün bir arkadaşım bana sürpriz yaptı, çok mutlu oldum",
    "Kendime zaman ayırmak için sinemaya gitmeyi planlıyorum",
    "Birkaç hafta sonra tatile çıkacağım, çok heyecanlıyım",
    "Bugün biraz alışveriş yapmayı düşünüyorum, yeni bir şeyler almak keyifli",
    "Küçük bir bahçem var, bitkilerime bakmak bana çok huzur veriyor",
    "Bir müzik festivali planlıyorum, çok heyecanlıyım",
    "Bugün yeni bir restoranda yemek yedim, tatları gerçekten harikaydı",
    "Spor yaparken dinlediğim müzikler çok motive edici",
    "Kendime yeni bir hedef koydum, bu hedefe ulaşmak için çalışıyorum",
    "Bugün çok güzel bir doğa manzarasına rastladım, fotoğraf çektim",
    "Sahilde yürüyüş yapmak, denizin sesini dinlemek çok rahatlatıcı",
    "Yeni bir şeyler öğrenmek, gün boyunca kendimi daha verimli hissettiriyor",
    "Bu hafta sonu için gezmeye gitmeyi planlıyorum",
    "Dün akşam bir konser izledim, çok keyif aldım",
    "Bugün dışarıda çok soğuk ama yine de keyifli bir gün",
    "Akşam yemeklerini sevdiklerimle paylaşmak çok keyifli",
    "Sonunda yeni bir projeyi bitirdim, gerçekten çok mutluyum",
    "Bazen sessiz bir ortamda yalnız kalıp düşünmek bana iyi geliyor",
    "Kahvemi içtikten sonra biraz yürüyüş yapmayı seviyorum",
    "Yeni bir uygulama keşfettim, çok faydalı bir şeydi",
    "Evde dinlenmek ve film izlemek için güzel bir gün",
    "Bugün iş yerinde çok verimli bir gün geçirdim",
    "Sonunda bir tatil planı yapabildim, çok heyecanlıyım",
    "Spor yaparak kendimi daha sağlıklı hissediyorum",
    "Sahilde gün batımını izlemek, rahatlamak için harika bir yol",
    "Bugün çok ilginç bir kitap okumaya başladım, konusu çok heyecanlı",
    "Kısa bir yürüyüş yapmak, tüm günü verimli geçirmek için iyi bir başlangıç",
    "Yeni bir film serisi izlemeye başladım, çok beğeniyorum",
    "Bugün mutfağa girip farklı yemekler yapmayı düşünüyorum",
    "Arkadaşım bana harika bir tavsiye verdi, çok işime yaradı",
    "Yeni bir etkinlik düzenlemeyi planlıyorum, çok heyecanlıyım",
    "Birkaç gün içinde yeni bir hobime başlamak istiyorum",
    "Bugün çok keyifli bir gün geçirdim, dışarıda çok fazla vakit geçirdim",
    "Kitapçıya gidip birkaç yeni kitap almak çok keyifliydi",
    "Hafta sonu gezmeye gitmeyi sabırsızlıkla bekliyorum",
    "Bugün güne dışarıda kahve içerek başladım, çok hoş bir gündü",
    "Kendime bir hedef koyup ona ulaşmak için çalışıyorum",
    "Geçen hafta yeni bir dil öğrenmeye başladım, ilerlemeye başladım",
    "Evde dinlenmek ve rahatlamak için akşamları film izlemek çok güzel",
    "Spor salonuna gidip çalışmak, kendimi çok iyi hissediyorum",
    "Bugün denizde yüzmeye gidip biraz dinlenmeyi planlıyorum",
    "Herkesin önerdiği bir kitabı okumaya başladım, çok ilginç",
    "Yeni bir müzik türü keşfettim, çok hoşuma gitti",
    "Bu hafta sonu gezmeye gitmek için birkaç arkadaşımı davet ettim",
    "Bugün çok güzel bir yürüyüş yaptım, doğa manzarası harikaydı",
    "Kendime bir gün ayırıp sadece rahatlayacağım",
    "Geceyi sakin bir ortamda kitap okuyarak geçirmeyi düşünüyorum",
    "Birkaç arkadaşım yeni bir etkinlik düzenliyor, katılmak için sabırsızlanıyorum",
    "Bugün çok keyifli bir alışveriş yaptım, yeni kıyafetler aldım",
    "Bugün çok güzel bir doğa fotoğrafı çektim, harika bir manzara",
    "Kendime yeni bir hedef koyup bu hedefe ulaşmak için çalışıyorum",
    "Bugün dışarıda uzun bir yürüyüş yaptım, gerçekten çok keyifliydi",
    "Arkadaşlarımla buluşmak için bir araya gelmeyi çok seviyorum",
    "Sabahları erken kalkıp güne başlamak bana çok iyi geliyor",
    "Kendime yeni bir hobim var, müzikle ilgilenmek çok rahatlatıcı",
    "Sonunda yeni bir tatil planı yapabildim, çok heyecanlıyım",
    "Bugün biraz alışveriş yapmayı düşünüyorum, yeni bir şeyler almak keyifli",
    "Bir arkadaşım bana sürpriz yaptı, çok mutlu oldum",
    "Geceyi açık havada geçirip yıldızları izlemek çok huzur verici",
    "Yeni bir şeyler öğrenmek çok keyifli, her gün bir şeyler katıyor hayatıma",
    "Kitap okumak, rahatlamak için harika bir yöntem",
    "Bugün sabah kahvemi içerken birkaç sayfa kitap okudum",
    "Kısa bir tatil planlıyorum, birkaç günlüğüne deniz kenarına gideceğim",
    "Dışarıda yürüyüş yapmak, günün stresini atmak için harika bir yöntem",
    "Evde vakit geçirmek için yeni bir film izlemeyi planlıyorum"
]



tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) +1

input_sequences =[]
for text in texts:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(1,len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        
max_sequence_length = max(len(x) for x in input_sequences)

input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding="pre")

X = input_sequences[:,:-1]
y = input_sequences[:,-1]

y = tf.keras.utils.to_categorical(y, num_classes = total_words)

model = Sequential()
model.add(Embedding(total_words, 50, input_length = X.shape[1]))

model.add(LSTM(100, return_sequences = False))

model.add(Dense(total_words, activation = "softmax"))

model.compile(optimizer = "adam",
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

model.fit(X,y,
          epochs=100,
          verbose=1)

def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen = max_sequence_length-1, padding="pre")
        predicted_probablities = model.predict(token_list, verbose =0)
        predicted_word_index = np.argmax(predicted_probablities, axis = -1)
        #tokenizer ile kelime indexinden asil kelimeyi bul
        predicted_word = tokenizer.index_word[predicted_word_index[0]]
        #tahmin edilen kelimeyi seed texte ekle
        seed_text = seed_text+ " " + predicted_word
    return seed_text

seed_text = "kahve"
print(generate_text(seed_text, 3))
