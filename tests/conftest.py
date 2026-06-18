import pytest
from src.clustering import build_tfidf


@pytest.fixture(scope="session")
def tfidf_vec():
    corpus = [
        "jalan sudirman nomor 5 jakarta pusat",
        "jalan pahlawan nomor 1 surabaya timur",
        "jalan asia afrika nomor 20 bandung",
        "gang mawar rt 1 rw 2 kelurahan gambir",
        "dusun bontomanai rt 2 rw 2",
        "dusun bontomanai rt 0 rw 0 kanjilo sunggu minasa",
    ]
    return build_tfidf(corpus)
