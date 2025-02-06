# NLP

</> Natural Language Processing - Doğal Dil İşleme

Bu repository, Doğal Dil İşleme (NLP) tekniklerini öğrenmek ve uygulamak isteyenler için çeşitli projeler ve örnekler içermektedir. Bu projelerde, metin verisini anlamak, temizlemek ve üzerinde farklı NLP modelleri ile işlem yapmak için temel ve ileri düzey teknikler bulunmaktadır.

1. Veri Temizleme ve Ön İşleme (Preprocessing)</br>
 Doğal dil işleme projelerinde, metin verisinin temizlenmesi ve işlenmesi, modelin başarısı için oldukça önemlidir. Bu aşama, metindeki gereksiz veya hatalı verileri temizlemek ve metni işleme hazırlamak için çeşitli teknikler içerir.

   Veri Temizleme Yöntemleri:

   - Boşlukları Düzenleme: Fazla boşlukları tek bir boşluğa indirgeme.
   - Büyük - Küçük Harf Çevirimi: Metindeki tüm harfleri küçük harfe çevirme.
   - Noktalama İşaretlerini Kaldırma: Metindeki noktalama işaretlerini kaldırma.
   - Özel Karakterleri Kaldırma: %, #, @, /, * gibi özel karakterleri temizleme.
   - Yazım Hatalarını Düzeltme: TextBlob kullanarak metindeki yazım hatalarını düzeltme.
   - HTML ve URL Etiketlerini Kaldırma: BeautifulSoup kullanarak HTML etiketlerini temizleme.
2. Tokenizasyon (Tokenization)</br>
Tokenizasyon, metni daha küçük parçalara (token'lara) ayırma işlemidir. Bu işlem, kelimeleri ve cümleleri anlamak için temel bir adımdır.

   - Kelime Tokenizasyonu (Word Tokenization): Metni kelimelere ayırma.
   - Cümle Tokenizasyonu (Sentence Tokenization): Metni cümlelere ayırma.
3. Stemming ve Lemmatization</br>
Kelime köklerine inmek, dil modellemeleri için önemlidir. Stemming, kelimelerin kökünü bulmaya çalışırken, Lemmatization daha doğru ve dil bilgisel anlamları dikkate alır.

   - Stemming: Kelimenin kökünü bulmak için kullanılan yöntem.
   - Lemmatization: Kelimenin kök halini doğru şekilde bulmak için dilbilgisel bilgileri kullanan hassas bir yöntem.
4. Stop Words (Durdurma Kelimeleri)</br>
Stop word'ler, genellikle anlam taşımayan ancak dilde sıklıkla karşılaşılan kelimelerdir. Bu kelimeleri temizleyerek metni daha anlamlı hale getirebiliriz.

   - Stop Word Temizleme: Stop word'leri metinden çıkarma.
     
Önemli NLP Yöntemleri ve Modelleri

1. Bag of Words (BoW)
Bag of Words (BoW), metindeki kelimeleri bir vektör haline getiren temel bir tekniktir. Bu yöntem, metindeki kelimelerin sırasını dikkate almaz; yalnızca kelimelerin varlığını ve sıklığını kullanır.

    Avantajları:

   - Kolay anlaşılır ve uygulanabilir.
   - Küçük veri setlerinde hızlı çalışır.
    Dezavantajları:

   - Kelimelerin sırasını dikkate almaz, bu da anlam kaybına yol açabilir.
   - Büyük veri setlerinde yüksek bellek ve işlem gücü gerektirir.
 2. TF-IDF (Term Frequency - Inverse Document Frequency)
 TF-IDF (Terim Frekansı - Ters Doküman Frekansı), kelimelerin metinlerdeki önemini ölçmek için yaygın olarak kullanılan bir tekniktir. Bu yöntem, her kelimenin metindeki frekansını dikkate alır ve o kelimenin metnin geri kalanındaki önemliğine göre ağırlık verir.

    - TF (Term Frequency): Bir kelimenin, dokümandaki tekrar sayısını ifade eder.
    - IDF (Inverse Document Frequency): Bir kelimenin, tüm dokümanlar arasında ne kadar nadir olduğunu ölçer.
   TF-IDF, sık görülen kelimeleri (örneğin "ve", "bir", "bu") önemsiz olarak kabul eder ve daha az görülen kelimelere daha fazla ağırlık verir.

 3. Özellik Çıkartma (Feature Extraction)
 Bu aşama, veriyi anlamak ve modelleri eğitmek için önemli olan özellikleri çıkartmaya yöneliktir. Özellik çıkarımı için yaygın yöntemlerden biri Word Embedding’dir.

    Word Embedding, kelimeleri vektörler halinde temsil eden bir tekniktir. Bu yöntemle, kelimeler arasındaki anlamsal ilişkiler daha etkili bir şekilde modellenebilir. Word2Vec, GloVe ve FastText gibi popüler embedding  teknikleri, kelimeleri yüksek boyutlu vektörlerle temsil eder.

    - Word2Vec: Kelimeleri vektör uzayına dönüştürerek kelimeler arasındaki anlam ilişkilerini modelleyen bir yöntemdir.
    - GloVe: Kelimelerin global istatistiksel özelliklerini kullanarak vektörleştirme yapar.
    - FastText: Kelimelerin alt birimlerine (n-gram'lar) bakarak daha anlamlı vektörler elde etmeyi amaçlar.
4. Probabilistic Models</br>
 Probabilistic Models, kelimeler arasındaki olasılık ilişkilerini kullanarak dilin yapısını analiz etmeye yönelik modellere odaklanır. Aşağıda bu kategorideki üç önemli model açıklanmıştır:

   1. N-Gram Models
    N-Gram modelleri, ardışık kelimeler arasındaki ilişkiyi anlamak için kullanılır. Bu tür modeller, metin oluşturmak veya tahmin yapmak için faydalıdır. N-Gram modellerinde kelimeler ardışık gruplar halinde ele alınır ve bu grupların olasılıkları hesaplanarak metinlerdeki ilişkiler anlaşılmaya çalışılır.

   - Bigram (2'li kelime grupları)
   - Trigram (3'lü kelime grupları)
   
   2. Hidden Markov Models (HMM)
   Hidden Markov Model (HMM), gözlemlerle gizli durumlar arasında olasılıksal bir ilişki kurar. Bu model, dildeki gizli yapıları anlamak için kullanılır. Örneğin, bir cümledeki kelimelerin hangi sözcük türlerine (isim, fiil, sıfat vb.) karşılık geldiğini tahmin etmek için HMM kullanılabilir.

   3. Maximum Entropy Models
  Maximum Entropy (MaxEnt) modelleri, veri üzerindeki belirsizliği minimize ederek en fazla entropiyi sağlayacak tahminler yapmak için kullanılır. Bu model, özellikle sınıflandırma problemlerinde etkilidir. Örneğin, metin sınıflandırma (duygu analizi) veya part-of-speech (POS) etiketleme gibi görevlerde kullanılabilir.


Doğal Dil İşleme Kütüphaneleri<br>
 - NLTK-Natural Language Toolkit kütüphanesi
 - Spacy
 - HuggingFace
 - TextBlob
 - BeautifulSoup
 - Gensim
 - fastText
 - Stanford toolkit (Glove)
 - Apache OpenNLP

