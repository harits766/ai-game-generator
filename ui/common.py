"""Komponen UI bersama: CSS, base URL, QR, sidebar dokumen."""
from __future__ import annotations

import io
from urllib.parse import urlparse

import segno
import streamlit as st

from core.documents import DocumentError, from_text, parse_bytes
from core.generator import cache_stats
from core.llm import _secret, status_label

OPTION_COLORS = ["#E11D48", "#2563EB", "#D97706", "#059669"]
OPTION_ICONS = ["▲", "◆", "●", "■"]

CSS = """
<style>
:root { --ink:#1F1534; --muted:#6B6385; --card:#FFFFFF; --line:#E4DDF7; --brand:#6D28D9; }
.block-container { padding-top: 1.6rem; max-width: 1100px; }
.hero { padding: 18px 20px; border-radius: 16px; color: #fff;
        background: linear-gradient(120deg, #6D28D9 0%, #DB2777 100%); margin-bottom: 14px; }
.hero h1 { color:#fff; margin:0; font-size: 1.7rem; }
.hero p { margin: 4px 0 0; opacity:.92; }
.card { background: var(--card); border:1px solid var(--line); border-radius: 14px; padding: 14px 16px;
        margin-bottom: 10px; }
.card h4 { margin: 0 0 4px; }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; background:#EFEAFE; color:var(--brand);
        font-size:.78rem; font-weight:600; margin-right:4px; }
.big-q { font-size: 1.45rem; font-weight: 700; line-height:1.35; margin: 6px 0 12px; }
.code { font-family: ui-monospace, monospace; font-size: 2.2rem; font-weight: 800; letter-spacing: .25rem;
        color: var(--brand); }
.score { font-size: 1.1rem; font-weight: 700; }
.muted { color: var(--muted); font-size: .9rem; }
.lb-row { display:flex; justify-content:space-between; padding:6px 10px; border-bottom:1px solid var(--line); }
.lb-row:nth-child(1) { font-weight:800; }
/* tombol jawaban besar berwarna ala game show */
div[class*="st-key-opt_"] button { min-height: 72px; font-size: 1.1rem; font-weight: 700; color:#fff;
        border:none; border-radius: 14px; white-space: normal; }
div[class*="st-key-opt_"] button:hover { filter: brightness(1.08); color:#fff; }
div[class*="st-key-opt_"][class*="_c0"] button { background:#E11D48; }
div[class*="st-key-opt_"][class*="_c1"] button { background:#2563EB; }
div[class*="st-key-opt_"][class*="_c2"] button { background:#D97706; }
div[class*="st-key-opt_"][class*="_c3"] button { background:#059669; }
@media (max-width: 640px) {
  .block-container { padding-left: 16px; padding-right: 16px; }
  .big-q { font-size: 1.2rem; }
  .code { font-size: 1.7rem; }
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def base_url() -> str:
    cfg = _secret("APP_URL")
    if cfg:
        return cfg.rstrip("/")
    try:
        u = urlparse(st.context.url)
        if u.scheme and u.netloc:
            return f"{u.scheme}://{u.netloc}"
    except Exception:
        pass
    return "http://localhost:8501"


def qr_png(url: str, scale: int = 6) -> bytes:
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="png", scale=scale, border=2)
    return buf.getvalue()


def sidebar_document():
    """Upload dokumen / topik; disimpan di session_state['doc'] untuk semua halaman."""
    with st.sidebar:
        st.markdown("### 📄 Dokumen sumber")
        up = st.file_uploader("PDF, DOCX, TXT, MD (maks 5 MB)", type=["pdf", "docx", "txt", "md"],
                              key="doc_upload")
        if up is not None and st.session_state.get("doc_name") != up.name:
            try:
                st.session_state["doc"] = parse_bytes(up.name, up.getvalue())
                st.session_state["doc_name"] = up.name
            except DocumentError as e:
                st.error(str(e))
        with st.expander("atau tempel teks / pakai contoh"):
            txt = st.text_area("Teks materi", height=120, key="doc_text")
            c1, c2 = st.columns(2)
            if c1.button("Pakai teks", use_container_width=True) and txt.strip():
                st.session_state["doc"] = from_text(txt, name="teks-tempel", doc_type="txt")
                st.session_state["doc_name"] = "teks-tempel"
            if c2.button("Contoh: sampah", use_container_width=True):
                from core.agent import sample_doc_for
                st.session_state["doc"] = sample_doc_for("sampah pilah")
                st.session_state["doc_name"] = "contoh"
        doc = st.session_state.get("doc")
        if doc:
            st.success(f"**{doc.name}** · {doc.doc_type.upper()} · {len(doc.text):,} karakter"
                       + (" (dipotong)" if doc.truncated else ""))
            if st.button("Hapus dokumen", use_container_width=True):
                st.session_state.pop("doc", None)
                st.session_state.pop("doc_name", None)
                st.rerun()
        else:
            st.caption("Tanpa dokumen, konten dibuat dari topik yang kamu tulis.")
        st.divider()
        st.markdown("### ⚙️ Mesin AI")
        st.caption(f"LLM: **{status_label()}**")
        st.toggle("Pakai LLM (matikan = mode offline)", value=True, key="use_llm")
        cs = cache_stats()
        st.caption(f"Cache konten: {cs['size']}/{cs['maxsize']} item · hit {cs['hit']} · "
                   f"miss {cs['miss']} · TTL {cs['ttl_sec'] // 60} menit")
    return st.session_state.get("doc")
