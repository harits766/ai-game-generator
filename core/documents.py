"""Ingest dokumen sumber (PDF, DOCX, TXT/MD, atau topik singkat) + retrieval sederhana (TF-IDF).

Retrieval dipakai agent (tool `read_document`) dan generator konten agar LLM hanya
melihat potongan paling relevan, bukan seluruh dokumen (hemat token free tier).
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import MAX_DOC_CHARS, MAX_UPLOAD_MB, SUPPORTED_DOC_TYPES

STOPWORDS_ID = set("""
ada adalah agar akan aku anda antara apa apabila atau bagai bagaimana bagi bahwa banyak
baru beberapa begitu belum benar berapa besar bisa boleh buat bukan cara dalam dan dapat dari
daripada demikian dengan di dia diri dua hal hanya harus hingga ia ialah ini itu jadi jika
juga kalau kami kamu karena ke kebanyakan kecil kemudian kepada ketika kita lagi lain lalu
lebih maka mana masih mau melalui memang mempunyai menjadi menurut mereka merupakan meski
misalnya mungkin namun nya oleh pada para pernah perlu pula saat saja salah sama sampai
sangat satu saya sebagai sebelum sedang sehingga sejak selain seluruh semua sendiri seperti
serta setelah setiap suatu sudah supaya tak tanpa tapi telah tentang terhadap termasuk tersebut
tetapi tidak tiga untuk walau yaitu yakni yang dll dsb pun kan tiap tersebut secara berbagai
the of and to in a is for on that with as are by be this from or an at it
""".split())


@dataclass
class Document:
    name: str
    doc_type: str
    text: str
    chunks: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def doc_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


class DocumentError(ValueError):
    pass


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)          # sambung kata terpotong
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_bytes(name: str, data: bytes) -> Document:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else "txt"
    if ext not in SUPPORTED_DOC_TYPES:
        raise DocumentError(f"Format .{ext} belum didukung. Pakai: {', '.join(SUPPORTED_DOC_TYPES)}.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise DocumentError(f"File lebih dari {MAX_UPLOAD_MB} MB (batas komputasi).")
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for p in reader.pages[:40]:                 # batas 40 halaman
            pages.append(p.extract_text() or "")
            if sum(map(len, pages)) > MAX_DOC_CHARS * 1.5:
                break
        text = "\n\n".join(pages)
        if len(text.strip()) < 50:
            raise DocumentError("PDF tidak berisi teks (kemungkinan hasil scan). OCR tidak didukung.")
    elif ext == "docx":
        import docx
        d = docx.Document(io.BytesIO(data))
        text = "\n".join(p.text for p in d.paragraphs)
    else:
        text = data.decode("utf-8", errors="ignore")
    return from_text(text, name=name, doc_type=ext)


def from_text(text: str, name: str = "topik", doc_type: str = "topic") -> Document:
    text = _clean(text)
    if not text:
        raise DocumentError("Dokumen kosong.")
    truncated = len(text) > MAX_DOC_CHARS
    text = text[:MAX_DOC_CHARS]
    return Document(name=name, doc_type=doc_type, text=text, chunks=chunk(text), truncated=truncated)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -•*#\t") for p in parts if len(p.strip()) > 0]


def chunk(text: str, size: int = 700) -> list[str]:
    out, buf = [], ""
    for s in split_sentences(text):
        if len(buf) + len(s) > size and buf:
            out.append(buf.strip())
            buf = ""
        buf += s + " "
    if buf.strip():
        out.append(buf.strip())
    return out or [text]


def retrieve(doc: Document, query: str, k: int = 3) -> list[str]:
    if len(doc.chunks) <= k:
        return doc.chunks
    vec = TfidfVectorizer(stop_words=list(STOPWORDS_ID), ngram_range=(1, 2))
    m = vec.fit_transform(doc.chunks + [query or doc.name])
    sims = (m[:-1] @ m[-1].T).toarray().ravel()
    if sims.max() == 0:   # query tidak cocok -> ambil potongan awal (biasanya inti materi)
        return doc.chunks[:k]
    return [doc.chunks[i] for i in np.argsort(-sims)[:k]]


GENERIC = set("""waktu hari sekitar ulang contoh antara lain hal cara bentuk jenis bagian
utama tingkat dasar sering mudah sulit sangat paling banyak sedikit orang tempat sesuai""".split())
_AFFIX = ("meng", "meny", "mem", "men", "me", "di", "ber", "ter", "per")


def _looks_verb(w: str) -> bool:
    return len(w) >= 7 and w.startswith(_AFFIX) and not w.startswith(("desa", "dinas", "data"))


def keywords(doc: Document, n: int = 20) -> list[str]:
    """Kata kunci khas (TF-IDF per kalimat). Kata yang muncul di >40% kalimat (mis. topik
    utama), kata umum, dan kata kerja berimbuhan dibuang agar pilihan jawaban lebih bermakna."""
    sents = split_sentences(doc.text)
    docs = sents if len(sents) > 3 else (doc.chunks or [doc.text])
    try:
        vec = TfidfVectorizer(stop_words=list(STOPWORDS_ID | GENERIC),
                              token_pattern=r"(?u)\b[^\W\d_]{4,}\b", lowercase=True)
        m = vec.fit_transform(docs)
    except ValueError:
        return []
    terms = vec.get_feature_names_out()
    df = np.asarray((m > 0).sum(axis=0)).ravel() / max(len(docs), 1)
    scores = np.asarray(m.max(axis=0).todense()).ravel()
    order = np.argsort(-scores)
    good = [terms[i] for i in order if df[i] <= 0.4 and not _looks_verb(terms[i])]
    rest = [terms[i] for i in order if terms[i] not in good]
    return (good + rest)[:n]
