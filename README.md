# Data Science Project 40 — Metin Benzerliği & Semantik Arama (TF-IDF + Cosine Similarity)

**Modül**: NLP (Doğal Dil İşleme) • **Süre**: 3-4 saat

## 🎯 Proje Senaryosu

Bir SaaS şirketinin **yardım merkezi / bilgi tabanı** (knowledge base) ekibinde **data scientist** olarak çalışıyorsun. Yüzlerce destek makalesi var ve iki problem can sıkıyor:

1. **Arama kötü.** Kullanıcı "uydu yörüngesine roket fırlatma" yazınca alakasız makaleler çıkıyor. Şirket, bir sorguya **anlamca en yakın** dokümanları getiren bir arama motoru istiyor.
2. **Tekrar eden makaleler var.** Farklı kişiler neredeyse aynı içeriği iki-üç kez yazmış. Bu **yakın-kopyaları (near-duplicate)** otomatik tespit etmek istiyorlar.

Senin görevin: anahtar kelime eşleşmesinin ötesine geçen, **mini bir semantik arama motoru** kurmak. Bir sorgu geldiğinde korpustaki tüm dokümanlarla **benzerliğini** ölçüp en alakalıları sıralayacak; ayrıca birbirine çok benzeyen doküman çiftlerini bulacak.

Bunun için **TF-IDF + Cosine Similarity** kullanacaksın. Bu projede **SINIFLANDIRMA YOK** — model eğitmiyoruz, etiket tahmin etmiyoruz. Bunun yerine her dokümanı bir **TF-IDF vektörüne** çevirip, vektörler arasındaki **açının kosinüsü** (cosine similarity) ile "ne kadar benzer" sorusunu cevaplıyoruz. Google'ın, Elasticsearch'ün, modern RAG sistemlerinin altında yatan **retrieval (bilgi getirme)** fikrinin en temel hali budur.

Korpus olarak **20 Newsgroups** veri setinden 4 farklı konuyu (uzay, otomobil, tıp, bilgisayar grafikleri) alacağız. Etiketleri sadece **doğrulama** için kullanacağız: TF-IDF gerçekten konuyu yakalıyorsa, **aynı kategoriden iki doküman**, farklı kategoriden iki dokümandan **daha benzer** çıkmalı (intra-similarity > inter-similarity).

Bu projede NLP dersinde öğrendiklerini birleştirip uygulayacaksın:
- ✅ **Gerçek dünya korpusu çekme** (`fetch_20newsgroups` — runtime cache)
- ✅ **Doküman-terim matrisi** (`TfidfVectorizer` — TF-IDF ağırlıkları)
- ✅ **Cosine similarity** (`sklearn.metrics.pairwise.cosine_similarity`)
- ✅ **Sorgu → vektör → en yakın dokümanlar** (semantik arama / retrieval)
- ✅ **Yakın-kopya (near-duplicate) tespiti** (eşik üstü çiftler)
- ✅ **En benzer doküman çifti** bulma
- ✅ **Kategori tutarlılığı** (intra vs inter benzerlik ile TF-IDF'i doğrulama)

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd data-science-project-40

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Dependencies
pip install -r requirements.txt

# Auto test runner (dosya değişince çalışır)
python watch.py

# Manuel test
pytest tests/test_question.py -v
```

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

Skorunun Kaizu hesabına yazılması için **`kaizu_config.py`** dosyasını aç ve **`USER_ID`** alanını kendi user_id'nle değiştir:

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 720 # ← Bu projeye ait, dokunma
```

User_id'ni Kaizu profilinden bulabilirsin (Profile → Settings → User ID).

Skor göndermek için tüm testleri toplu çalıştırmalısın:

```bash
python tests/test_question.py
```

Bu komut tüm testleri çalıştırır, **passed/total oranını otomatik Kaizu'ya gönderir**. Geliştirme sırasında `pytest -v` kullanmaya devam edebilirsin (skor göndermez).

## 📚 Dataset — 20 Newsgroups (sklearn)

### Kaynak
- **20 Newsgroups** (Ken Lang, 1995) — sklearn ile birlikte gelir.
- Detay: https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset

### Veri Çekme Yöntemi (önemli)
Veri seti **repo'da YOK** — `fetch_corpus()` fonksiyonun veriyi **runtime'da** sklearn ile çeker:

1. `fetch_20newsgroups(subset='train', categories=[...], remove=('headers','footers','quotes'))`
2. İlk çağrıda ~14 MB'lık arşiv kullanıcının **`~/scikit_learn_data`** klasörüne iner ve cache'lenir.
3. Sonraki tüm çağrılar diskten okur — internet sadece ilk seferde gerekir.
4. İlk **~1000 dokümanı** korpus olarak alırız (`limit`).

> **Not — `remove`:** `('headers', 'footers', 'quotes')` ile e-posta başlıkları, imza/footer ve alıntı satırları atılır. Böylece benzerlik **gerçek metin içeriğinden** hesaplanır, sızıntı (leakage) yapan meta-bilgilerden değil.

### Kullanılan Kategoriler (4 konu)
| Kategori | Konu |
|----------|------|
| `sci.space` | Uzay, roket, NASA, yörünge |
| `rec.autos` | Otomobil, motor, araba |
| `sci.med` | Tıp, sağlık, hastalık |
| `comp.graphics` | Bilgisayar grafikleri, render, 3D |

### Bu projede SINIFLANDIRMA YOK
Etiketleri **model eğitmek için kullanmıyoruz**. Sadece **doğrulama** için: TF-IDF konuyu gerçekten yakalıyorsa, aynı kategoriden dokümanlar birbirine daha benzer (yüksek intra-similarity), farklı kategoridekiler daha az benzer (düşük inter-similarity) çıkmalı.

## 🔬 TF-IDF + Cosine Similarity — Nasıl Çalışır?

1. **TF-IDF**: Her dokümanı bir sayı vektörüne çevirir. Bir kelimenin ağırlığı, o dokümanda ne sıklıkta geçtiğine (**TF**) ve korpus genelinde ne kadar nadir olduğuna (**IDF**) bağlıdır. "the", "is" gibi her yerde geçen kelimeler düşük ağırlık alır; konuya özel kelimeler ("rocket", "engine") yüksek.
2. **Cosine Similarity**: İki vektör arasındaki **açının kosinüsü**. 1.0 = aynı yöne bakıyor (çok benzer), 0.0 = dik (alakasız). TF-IDF vektörleri negatif olmadığı için skor **[0, 1]** aralığındadır.
3. **Retrieval (arama)**: Sorguyu da aynı vektör uzayına atıp, korpustaki her dokümanla cosine benzerliğini hesaplar, en yüksekten sıralarız → semantik arama sonucu.

## 📋 Görevler (`tasks/task_manager.py`)

`task_manager.py` dosyasındaki **12 fonksiyonu** sırayla doldur. Her task altta testler pass olana kadar düzenlenmeli.

1. **`fetch_corpus(categories=..., limit=1000)`** — 20 Newsgroups'tan korpusu çek → `(docs, labels, target_names)`
2. **`explore_corpus(docs, labels, target_names)`** — `{n_docs, n_categories, avg_doc_length, docs_per_category}`
3. **`build_tfidf_matrix(docs)`** — `TfidfVectorizer(stop_words='english', max_features=5000)` → `(vectorizer, tfidf_matrix)`
4. **`query_to_vector(vectorizer, query)`** — sorgu string'ini TF-IDF vektörüne çevir (1×V)
5. **`compute_similarities(query_vec, tfidf_matrix)`** — sorgu vs her doküman cosine benzerliği (1D array)
6. **`most_similar_docs(vectorizer, tfidf_matrix, query, k=5)`** — en benzer k doküman `[{index, score}]`
7. **`pairwise_similarity(tfidf_matrix, i, j)`** — i ve j dokümanları arası cosine benzerlik
8. **`find_near_duplicates(tfidf_matrix, threshold=0.5)`** — eşik üstü `(i, j, score)` çiftleri (i<j)
9. **`most_similar_pair(tfidf_matrix)`** — tek en benzer farklı doküman çifti `(i, j, score)`
10. **`search(vectorizer, tfidf_matrix, docs, query, k=3)`** — `[{index, score, preview}]` kullanılabilir arama sonucu
11. **`category_coherence(tfidf_matrix, labels)`** — `{intra, inter}` ortalama benzerlikler (intra > inter beklenir)
12. **`run_pipeline()`** — uçtan uca akış, özet dict dön

## 🎓 Öğrenme Hedefleri

Bu projeyi bitirdiğinde:
- [x] Gerçek bir **metin korpusunu** çekip keşfedebileceksin (`fetch_20newsgroups`)
- [x] **TF-IDF** ile doküman-terim matrisi kurabileceksin
- [x] **Cosine similarity** ile vektörler arası benzerliği ölçebileceksin
- [x] Bir sorguyu vektörleştirip **en yakın dokümanları getirebileceksin** (semantik arama)
- [x] **Yakın-kopya** dokümanları eşik bazlı tespit edebileceksin
- [x] **Intra vs inter** benzerlik karşılaştırmasıyla TF-IDF'in konuyu yakaladığını gösterebileceksin

## 🧪 Testler

Test dosyası: `tests/test_question.py` (15 test)

Tümü pass olmalı:
- Korpus çekme (ilk test sklearn ile indirir) + `limit` uygulanıyor mu
- TF-IDF matris şekli (`n_docs × ≤5000`)
- Cosine skorları **[0, 1]** aralığında mı
- Bir dokümanın **kendisiyle benzerliği ≈ 1.0** mı
- `most_similar_docs` k adet ve **azalan** sıralı sonuç dönüyor mu
- **Uzay sorgusunun** en benzer sonuçları çoğunlukla `sci.space` mi
- **Near-duplicate** tespiti eşik (`≥threshold`) üstü çiftler dönüyor mu
- **Kategori tutarlılığı**: `intra_sim > inter_sim` mi

## 📊 Beklenen Sonuçlar

```
Doküman sayısı            : ~1000
Uzay sorgusu top kategori : sci.space
Intra-kategori benzerlik  : inter'den belirgin yüksek
Inter-kategori benzerlik  : intra'dan düşük
En benzer çift skoru      : yüksek (0.5+ — yakın-kopya / aynı konu)
```

## 💡 İpuçları

- **İlk testte internet** gerekli (sklearn arşivi indirir). Sonraki testler `~/scikit_learn_data`'dan okur.
- `cosine_similarity` `sklearn.metrics.pairwise`'ten gelir; sparse TF-IDF matrisini doğrudan kabul eder.
- `query_to_vector`'da sorgu için **`fit_transform` değil `transform`** kullan (korpusun kelime dağarcığına göre).
- `compute_similarities` 2B değil **1B** array dönmeli → `cosine_similarity(qv, M).ravel()`.
- `np.argsort(sims)[::-1][:k]` → en büyük k benzerliğin indeksleri (azalan).
- Self-similarity tam **1.0** çıkmalı (aynı vektör, sıfır açı).
- `find_near_duplicates`: tüm matris için `cosine_similarity(M)` (N×N) hesaplayıp üst üçgende (i<j) eşik üstü çiftleri topla.
- `most_similar_pair`: N×N benzerlik matrisinin **köşegenini** ele (her doküman kendine 1.0 benzer), sonra max'ı bul.
- `category_coherence`: aynı etiketli çiftler → intra, farklı etiketli çiftler → inter. **intra > inter** beklenir.

## 🚫 Dikkat

- `tests/test_question.py` dosyasını **değiştirme**
- `random_state=42` değerini değiştirme (korpus sırası değişir, testler fail olabilir)
- `_solution/` klasörü yok (DB'de saklanır, dersin haftası geçince açılır)
- `data/` ve `scikit_learn_data/` repo'ya **gitmez** (.gitignore'da exclude)
- Dokunabileceğin **2 dosya**: `tasks/task_manager.py` (kodu yaz) + `kaizu_config.py` (sadece USER_ID)
