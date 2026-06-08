import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    fetch_corpus, explore_corpus, build_tfidf_matrix, query_to_vector,
    compute_similarities, most_similar_docs, pairwise_similarity,
    find_near_duplicates, most_similar_pair, search,
    category_coherence, run_pipeline,
)


# ──────────────────────────────────────────────────────
# Modül-seviye cache — testler arası tekrar indirme/vektörize yok
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def corpus():
    """İlk testte sklearn ile indirir, sonraki tüm testler cache'den okur."""
    return fetch_corpus()


@pytest.fixture(scope="module")
def tfidf(corpus):
    docs, labels, target_names = corpus
    vectorizer, tfidf_matrix = build_tfidf_matrix(docs)
    return vectorizer, tfidf_matrix


SPACE_QUERY = "NASA rocket launch to Mars orbit Apollo astronaut shuttle"


# 1. fetch_corpus
def test_fetch_corpus_structure(corpus):
    docs, labels, target_names = corpus
    assert isinstance(docs, list)
    assert isinstance(labels, list)
    assert isinstance(target_names, list)
    assert len(docs) == len(labels)
    assert len(docs) > 0
    assert all(isinstance(d, str) for d in docs[:5])
    assert set(target_names) == {'sci.space', 'rec.autos', 'sci.med', 'comp.graphics'}


# 2. fetch_corpus — limit uygulanıyor mu
def test_fetch_corpus_limit():
    docs, labels, target_names = fetch_corpus(limit=200)
    assert len(docs) == 200
    assert len(labels) == 200


# 3. explore_corpus
def test_explore_corpus(corpus):
    docs, labels, target_names = corpus
    info = explore_corpus(docs, labels, target_names)
    assert set(info.keys()) >= {
        'n_docs', 'n_categories', 'avg_doc_length', 'docs_per_category'
    }
    assert info['n_docs'] == len(docs)
    assert info['n_categories'] == 4
    assert info['avg_doc_length'] > 0
    assert sum(info['docs_per_category'].values()) == info['n_docs']


# 4. build_tfidf_matrix — şekil
def test_build_tfidf_matrix_shape(corpus, tfidf):
    docs, labels, target_names = corpus
    vectorizer, tfidf_matrix = tfidf
    assert tfidf_matrix.shape[0] == len(docs)
    # max_features=5000 → terim sayısı 5000'i geçmemeli
    assert tfidf_matrix.shape[1] <= 5000


# 5. query_to_vector — 1 satır, korpusla aynı terim boyutu
def test_query_to_vector(tfidf):
    vectorizer, tfidf_matrix = tfidf
    qv = query_to_vector(vectorizer, "space rocket launch")
    assert qv.shape[0] == 1
    assert qv.shape[1] == tfidf_matrix.shape[1]


# 6. compute_similarities — [0,1] aralığı, doğru uzunluk
def test_compute_similarities_range(tfidf):
    vectorizer, tfidf_matrix = tfidf
    qv = query_to_vector(vectorizer, SPACE_QUERY)
    sims = compute_similarities(qv, tfidf_matrix)
    assert isinstance(sims, np.ndarray)
    assert sims.shape[0] == tfidf_matrix.shape[0]
    # TF-IDF negatif olmadığı için cosine benzerlik [0,1] aralığında
    assert sims.min() >= -1e-9
    assert sims.max() <= 1 + 1e-9


# 7. pairwise_similarity — self-similarity ≈ 1.0
def test_self_similarity_is_one(tfidf):
    vectorizer, tfidf_matrix = tfidf
    s = pairwise_similarity(tfidf_matrix, 0, 0)
    assert abs(s - 1.0) < 1e-6


# 8. pairwise_similarity — simetrik ve [0,1]
def test_pairwise_similarity_symmetric(tfidf):
    vectorizer, tfidf_matrix = tfidf
    s_ij = pairwise_similarity(tfidf_matrix, 3, 7)
    s_ji = pairwise_similarity(tfidf_matrix, 7, 3)
    assert abs(s_ij - s_ji) < 1e-9
    assert 0 <= s_ij <= 1 + 1e-9


# 9. most_similar_docs — k adet, azalan sıralı, [0,1]
def test_most_similar_docs_sorted(tfidf):
    vectorizer, tfidf_matrix = tfidf
    hits = most_similar_docs(vectorizer, tfidf_matrix, SPACE_QUERY, k=5)
    assert len(hits) == 5
    scores = [h['score'] for h in hits]
    # Azalan sırada olmalı
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 1 + 1e-9 for s in scores)
    assert all('index' in h and 'score' in h for h in hits)


# 10. Uzay sorgusu → sonuçların çoğunluğu sci.space
def test_space_query_returns_space_docs(corpus, tfidf):
    docs, labels, target_names = corpus
    vectorizer, tfidf_matrix = tfidf
    space_idx = target_names.index('sci.space')
    hits = most_similar_docs(vectorizer, tfidf_matrix, SPACE_QUERY, k=5)
    hit_labels = [labels[h['index']] for h in hits]
    n_space = sum(1 for l in hit_labels if l == space_idx)
    # En benzer 5 dokümanın çoğunluğu sci.space olmalı
    assert n_space >= 3


# 11. search — preview ve alanlar
def test_search_result_format(corpus, tfidf):
    docs, labels, target_names = corpus
    vectorizer, tfidf_matrix = tfidf
    results = search(vectorizer, tfidf_matrix, docs, SPACE_QUERY, k=3)
    assert len(results) == 3
    for r in results:
        assert set(r.keys()) >= {'index', 'score', 'preview'}
        assert isinstance(r['preview'], str)
        assert len(r['preview']) <= 80
        assert r['preview'] == docs[r['index']][:80]


# 12. find_near_duplicates — eşik üstü çiftler
def test_find_near_duplicates(tfidf):
    vectorizer, tfidf_matrix = tfidf
    threshold = 0.5
    pairs = find_near_duplicates(tfidf_matrix, threshold=threshold)
    assert isinstance(pairs, list)
    for i, j, score in pairs:
        assert i < j
        assert score >= threshold
    # Düşük eşik daha çok (en azından eşit) çift bulmalı
    more = find_near_duplicates(tfidf_matrix, threshold=0.3)
    assert len(more) >= len(pairs)


# 13. most_similar_pair — i<j, [0,1]
def test_most_similar_pair(tfidf):
    vectorizer, tfidf_matrix = tfidf
    i, j, score = most_similar_pair(tfidf_matrix)
    assert i < j
    assert 0 <= score <= 1 + 1e-9
    # Bu çiftin skoru ortalama benzerlikten yüksek olmalı
    assert score > 0


# 14. category_coherence — intra > inter (TF-IDF konuyu yakalıyor)
def test_category_coherence(tfidf, corpus):
    docs, labels, target_names = corpus
    vectorizer, tfidf_matrix = tfidf
    coh = category_coherence(tfidf_matrix, labels)
    assert set(coh.keys()) >= {'intra', 'inter'}
    # Aynı kategori dokümanları, farklı kategoriden daha benzer olmalı
    assert coh['intra'] > coh['inter']


# 15. run_pipeline — uçtan uca
def test_run_pipeline_full():
    result = run_pipeline()
    assert set(result.keys()) >= {
        'n_docs', 'top_result_for_space_query',
        'intra_sim', 'inter_sim', 'most_similar_pair_score',
    }
    assert result['n_docs'] > 0
    # Uzay sorgusunun en benzer dokümanı sci.space kategorisinde olmalı
    assert result['top_result_for_space_query'] == 'sci.space'
    # TF-IDF konu tutarlılığını yakalar
    assert result['intra_sim'] > result['inter_sim']
    assert 0 <= result['most_similar_pair_score'] <= 1 + 1e-9


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
