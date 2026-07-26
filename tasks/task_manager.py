"""
DS-40 — Metin Benzerliği & Semantik Arama (TF-IDF + Cosine Similarity)
Bir yardım merkezi / bilgi tabanı ekibinde data scientist'sin. Bir sorguya
anlamca en yakın dokümanları getiren mini bir "semantik arama" motoru kuruyorsun
ve birbirine çok benzeyen (yakın-kopya) makaleleri tespit ediyorsun.

Bu projede SINIFLANDIRMA YOK — model eğitmiyoruz. Her dokümanı TF-IDF vektörüne
çevirip cosine_similarity ile "ne kadar benzer" sorusunu cevaplıyoruz. Etiketleri
sadece doğrulama için kullanıyoruz (intra > inter benzerlik).

Her fonksiyonun pass kısmını doldur. Testleri çalıştır, hepsi geçene kadar
iterate et: `python watch.py` veya `pytest tests/test_question.py -v`
"""

import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Korpusu çek — 20 Newsgroups
def fetch_corpus(categories=None, limit=1000):
    """
    20 Newsgroups veri setinden bir metin korpusu çek (cache'li, runtime).

    Akış:
    1. fetch_20newsgroups ile train subset'ini çek:
       fetch_20newsgroups(subset='train', categories=categories,
                          remove=('headers', 'footers', 'quotes'),
                          shuffle=True, random_state=42)
       - remove=(...) → e-posta başlık/footer/alıntı satırlarını atar
         (benzerlik gerçek içerikten hesaplansın, leakage olmasın)
       - random_state=42 → tekrarlanabilir doküman sırası (DEĞİŞTİRME)
    2. İlk `limit` dokümanı al (korpusu küçük tut): news.data[:limit]
    3. Etiketleri int listesine çevir: news.target[:limit]
    4. target_names'i listeye çevir.

    Kullanılan kategoriler (categories=None ise default):
        ['sci.space', 'rec.autos', 'sci.med', 'comp.graphics']

    Args:
        categories: çekilecek kategori listesi (None ise default 4 kategori)
        limit: korpusa alınacak maksimum doküman sayısı (default 1000)

    Returns:
        tuple: (docs:list[str], labels:list[int], target_names:list[str])

    İpucu:
    - from sklearn.datasets import fetch_20newsgroups
    - İlk çağrı ~14 MB arşivi ~/scikit_learn_data'ya indirir (internet gerekli),
      sonraki çağrılar diskten okur.
    - docs = list(news.data[:limit])
    - labels = [int(t) for t in news.target[:limit]]
    """
    
    if categories is None:
        categories = [
            "sci.space",
            "rec.autos",
            "sci.med",
            "comp.graphics"
        ]

    news = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42
    )

    docs = list(news.data[:limit])
    labels = [int(t) for t in news.target[:limit]]
    target_names = list(news.target_names)

    return docs, labels, target_names


# 2. Korpusu keşfet
def explore_corpus(docs, labels, target_names):
    """
    Korpusun temel istatistiklerini üret.

    Args:
        docs: doküman metinleri (list[str])
        labels: her dokümanın kategori indeksi (list[int])
        target_names: kategori adları (list[str])

    Returns:
        dict: {
            'n_docs': int (doküman sayısı),
            'n_categories': int (kategori sayısı),
            'avg_doc_length': float (ortalama doküman uzunluğu — KELİME cinsinden),
            'docs_per_category': dict (kategori_adı -> doküman sayısı)
        }

    İpucu:
    - Kelime sayısı: len(doc.split())
    - avg_doc_length = np.mean([len(d.split()) for d in docs])
    - docs_per_category: her idx için labels içinde kaç kez geçtiğini say
      (enumerate(target_names) + sum(1 for l in labels if l == idx))
    """
    
    return {
        "n_docs": len(docs),
        "n_categories": len(target_names),
        "avg_doc_length": np.mean([len(doc.split()) for doc in docs]),
        "docs_per_category": {
            category: sum(1 for label in labels if label == idx)
            for idx, category in enumerate(target_names)
        }
    }


# 3. TF-IDF doküman-terim matrisi kur
def build_tfidf_matrix(docs):
    """
    TfidfVectorizer ile doküman-terim matrisi oluştur.

    - TfidfVectorizer(stop_words='english', max_features=5000)
      → stop_words='english': 'the', 'is', 'a' gibi anlamsız kelimeleri eler
      → max_features=5000: en sık 5000 terimi tutar (matris boyutu kontrolü)
    - fit_transform(docs) → sparse TF-IDF matrisi (n_docs × n_terms)

    Args:
        docs: doküman metinleri (list[str])

    Returns:
        tuple: (vectorizer, tfidf_matrix)
            - vectorizer: fit edilmiş TfidfVectorizer
            - tfidf_matrix: scipy sparse matris (n_docs × n_terms)

    İpucu:
    - from sklearn.feature_extraction.text import TfidfVectorizer
    - vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    - tfidf_matrix = vectorizer.fit_transform(docs)
    """
    
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer, tfidf_matrix


# 4. Sorgu metnini TF-IDF vektörüne çevir
def query_to_vector(vectorizer, query):
    """
    Bir sorgu string'ini, korpusun kelime dağarcığına göre TF-IDF vektörüne çevir.

    DİKKAT: Burada fit_transform DEĞİL, transform kullan. Sorguyu zaten fit
    edilmiş vectorizer'ın bildiği terimlere göre vektörleştiriyoruz.

    Args:
        vectorizer: build_tfidf_matrix'ten dönen fit edilmiş TfidfVectorizer
        query: sorgu metni (str)

    Returns:
        1×V şeklinde sparse TF-IDF vektörü (V = korpus terim sayısı)

    İpucu:
    - transform bir liste bekler → vectorizer.transform([query])
    """
    
    return vectorizer.transform([query])


# 5. Sorgu ile tüm dokümanların cosine benzerliği
def compute_similarities(query_vec, tfidf_matrix):
    """
    Sorgu vektörü ile korpustaki HER dokümanın cosine benzerliğini hesapla.

    Args:
        query_vec: 1×V sorgu TF-IDF vektörü (query_to_vector çıktısı)
        tfidf_matrix: n_docs × V doküman-terim matrisi

    Returns:
        np.ndarray: 1B array, uzunluğu n_docs — her dokümanın benzerlik skoru.
                    TF-IDF negatif olmadığı için skorlar [0, 1] aralığında.

    İpucu:
    - from sklearn.metrics.pairwise import cosine_similarity
    - cosine_similarity(query_vec, tfidf_matrix) → 1×n_docs (2B)
    - .ravel() ile 1B array'e düzleştir
    """
    
    return cosine_similarity(query_vec, tfidf_matrix).ravel()


# 6. En benzer k doküman
def most_similar_docs(vectorizer, tfidf_matrix, query, k=5):
    """
    Sorguya en benzer k dokümanı azalan benzerlik sırasıyla döndür.

    Args:
        vectorizer: fit edilmiş TfidfVectorizer
        tfidf_matrix: doküman-terim matrisi
        query: sorgu metni (str)
        k: kaç sonuç döndürülecek (default 5)

    Returns:
        list[dict]: [{'index': int, 'score': float}, ...]
                    en yüksek benzerlikten düşüğe sıralı, score ∈ [0, 1]

    İpucu:
    - query_vec = query_to_vector(vectorizer, query)
    - sims = compute_similarities(query_vec, tfidf_matrix)
    - top_idx = np.argsort(sims)[::-1][:k]  # büyükten küçüğe ilk k indeks
    - [{'index': int(i), 'score': float(sims[i])} for i in top_idx]
    """
    
    query_vec = query_to_vector(vectorizer, query)
    sims = compute_similarities(query_vec, tfidf_matrix)

    top_idx = np.argsort(sims)[::-1][:k]

    return [
        {
            "index": int(i),
            "score": float(sims[i])
        }
        for i in top_idx
    ]


# 7. İki doküman arası cosine benzerlik
def pairwise_similarity(tfidf_matrix, i, j):
    """
    i. ve j. dokümanlar arasındaki cosine benzerliği döndür.

    Args:
        tfidf_matrix: doküman-terim matrisi
        i, j: doküman indeksleri (int)

    Returns:
        float: cosine benzerlik ∈ [0, 1]. i == j ise ~1.0 (kendisiyle).

    İpucu:
    - cosine_similarity(tfidf_matrix[i], tfidf_matrix[j]) → 1×1 matris
    - float(... [0, 0]) ile skaler değeri al
    """
    
    return float(cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0, 0])


# 8. Yakın-kopya (near-duplicate) çiftleri
def find_near_duplicates(tfidf_matrix, threshold=0.5):
    """
    Birbirine çok benzeyen doküman çiftlerini (yakın-kopya) bul.

    Tüm doküman çiftleri arasındaki cosine benzerliği hesapla; benzerliği
    threshold'dan büyük/eşit olan (i, j) çiftlerini topla. Sadece i < j
    çiftlerini al (her çift bir kez, kendisiyle karşılaştırma yok).

    Args:
        tfidf_matrix: doküman-terim matrisi
        threshold: bu eşiğin üstündeki çiftler "yakın-kopya" (default 0.5)

    Returns:
        list[tuple]: [(i, j, score), ...] — i < j, score >= threshold

    İpucu:
    - sims = cosine_similarity(tfidf_matrix)  # N×N matris
    - İç içe döngü: for i in range(n): for j in range(i+1, n):
        score = float(sims[i, j]); if score >= threshold: pairs.append((i, j, score))
    """
    
    sims = cosine_similarity(tfidf_matrix)
    n = sims.shape[0]

    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            score = float(sims[i, j])

            if score >= threshold:
                pairs.append((i, j, score))

    return pairs


# 9. En benzer doküman çifti
def most_similar_pair(tfidf_matrix):
    """
    Korpustaki birbirine EN benzer (farklı) doküman çiftini bul.

    DİKKAT: Her doküman kendisiyle 1.0 benzer — köşegeni ele (yoksa hep (i, i)
    çıkar). Köşegeni -1 yapıp matrisin en büyük değerini bul.

    Args:
        tfidf_matrix: doküman-terim matrisi

    Returns:
        tuple: (i, j, score) — i < j, score = o çiftin cosine benzerliği

    İpucu:
    - sims = cosine_similarity(tfidf_matrix)
    - np.fill_diagonal(sims, -1.0)  # kendisiyle karşılaştırmayı ele
    - i, j = np.unravel_index(np.argmax(sims), sims.shape)
    - i, j = int(i), int(j); i > j ise swap (i < j olsun)
    """
    sims = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sims, -1.0)
    i, j = np.unravel_index(np.argmax(sims), sims.shape)
    i, j = int(i), int(j)

    if i > j:
        i, j = j, i

    return i, j, float(sims[i, j])


# 10. Kullanılabilir arama sonucu
def search(vectorizer, tfidf_matrix, docs, query, k=3):
    """
    Kullanıcıya gösterilebilecek arama sonucu üret: index + skor + önizleme.

    Args:
        vectorizer: fit edilmiş TfidfVectorizer
        tfidf_matrix: doküman-terim matrisi
        docs: orijinal doküman metinleri (önizleme için)
        query: sorgu metni (str)
        k: kaç sonuç (default 3)

    Returns:
        list[dict]: [{'index': int, 'score': float, 'preview': str}, ...]
                    preview = dokümanın ilk 80 karakteri

    İpucu:
    - hits = most_similar_docs(vectorizer, tfidf_matrix, query, k=k)
    - Her hit için preview = docs[hit['index']][:80]
    """
    
    hits = most_similar_docs(vectorizer, tfidf_matrix, query, k=k)

    return [
        {
            "index": hit["index"],
            "score": hit["score"],
            "preview": docs[hit["index"]][:80]
        }
        for hit in hits
    ]


# 11. Kategori tutarlılığı — intra vs inter benzerlik
def category_coherence(tfidf_matrix, labels):
    """
    TF-IDF'in konuyu yakalayıp yakalamadığını ölç.

    Tüm doküman çiftlerini gez:
    - AYNI kategoriden iki doküman → intra benzerlik listesine
    - FARKLI kategoriden iki doküman → inter benzerlik listesine
    Her iki listenin ortalamasını al. Beklenti: intra > inter
    (aynı konudaki dokümanlar birbirine daha benzer olmalı).

    Args:
        tfidf_matrix: doküman-terim matrisi
        labels: her dokümanın kategori indeksi (list[int])

    Returns:
        dict: {'intra': float, 'inter': float}

    İpucu:
    - sims = cosine_similarity(tfidf_matrix)
    - labels = np.asarray(labels)
    - for i in range(n): for j in range(i+1, n):
        labels[i] == labels[j] ise intra'ya, değilse inter'e ekle
    - intra = np.mean(intra_vals), inter = np.mean(inter_vals)
    """
    
    sims = cosine_similarity(tfidf_matrix)
    labels = np.asarray(labels)

    n = sims.shape[0]

    intra_vals = []
    inter_vals = []

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra_vals.append(sims[i, j])
            else:
                inter_vals.append(sims[i, j])

    return {
        "intra": float(np.mean(intra_vals)),
        "inter": float(np.mean(inter_vals))
    }


# 12. Uçtan uca pipeline
def run_pipeline():
    """
    Uçtan uca akış:
    1. fetch_corpus → (docs, labels, target_names)
    2. build_tfidf_matrix → (vectorizer, tfidf_matrix)
    3. Uzay temalı sorgu ile arama yap, en benzer dokümanın kategorisini al:
       query = "NASA rocket launch to Mars orbit"
       most_similar_docs(..., k=1) → top_idx → target_names[labels[top_idx]]
       (sci.space çıkmasını bekliyoruz)
    4. category_coherence → intra / inter benzerlik
    5. most_similar_pair → en benzer çiftin skoru

    Returns:
        dict: {
            'n_docs': int,
            'top_result_for_space_query': str (uzay sorgusu top dokümanın kategorisi),
            'intra_sim': float,
            'inter_sim': float,
            'most_similar_pair_score': float
        }
    """
    
    # 1. fetch_corpus → (docs, labels, target_names)
    docs, labels, target_names = fetch_corpus()
    
    # 2. build_tfidf_matrix → (vectorizer, tfidf_matrix)
    vectorizer, tfidf_matrix = build_tfidf_matrix(docs)
    
    # 3. Uzay temalı sorgu ile arama yap, en benzer dokümanın kategorisini al:
    #    query = "NASA rocket launch to Mars orbit"
    #    most_similar_docs(..., k=1) → top_idx → target_names[labels[top_idx]]
    #    (sci.space çıkmasını bekliyoruz)
    query = "NASA rocket launch to Mars orbit"
    
    top_result = most_similar_docs(
        vectorizer,
        tfidf_matrix,
        query,
        k=1
    )

    top_idx = top_result[0]["index"]
    top_category = target_names[labels[top_idx]]
    
    # 4. category_coherence → intra / inter benzerlik
    coherence = category_coherence(tfidf_matrix, labels)
    
    # 5. most_similar_pair → en benzer çiftin skoru
    _, _, pair_score = most_similar_pair(tfidf_matrix)
    
    return {
        "n_docs": len(docs),
        "top_result_for_space_query": top_category,
        "intra_sim": coherence["intra"],
        "inter_sim": coherence["inter"],
        "most_similar_pair_score": pair_score,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 Pipeline Sonuçları:")
    print(f"  Doküman sayısı            : {result['n_docs']}")
    print(f"  Uzay sorgusu top kategori : {result['top_result_for_space_query']}")
    print(f"  Intra-kategori benzerlik  : {result['intra_sim']:.4f}")
    print(f"  Inter-kategori benzerlik  : {result['inter_sim']:.4f}")
    print(f"  En benzer çift skoru      : {result['most_similar_pair_score']:.4f}")
