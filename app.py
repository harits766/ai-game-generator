"""AI Game Generator: entry point Streamlit.

- URL dengan ?room=KODE        -> tampilan peserta
- URL dengan ?room=KODE&role=host&key=... -> tampilan host
- selain itu                   -> studio (chat agent, buat room, custom, tren, ground truth, model)
"""
import streamlit as st

st.set_page_config(page_title="AI Game Generator", page_icon="🎲", layout="wide",
                   initial_sidebar_state="auto")

from ui.common import inject_css, sidebar_document  # noqa: E402

inject_css()

if st.query_params.get("room"):
    from ui.room import render_room
    st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl']"
                "{display:none}</style>", unsafe_allow_html=True)
    render_room()
else:
    from ui import studio
    sidebar_document()
    pages = [
        st.Page(studio.page_chat, title="Game Master AI", icon="🤖", default=True),
        st.Page(studio.page_manual, title="Buat Room Manual", icon="🛠️", url_path="buat"),
        st.Page(studio.page_custom, title="Game Custom", icon="🧪", url_path="custom"),
        st.Page(studio.page_trends, title="Tren & Library", icon="📈", url_path="tren"),
        st.Page(studio.page_ground_truth, title="Ground Truth", icon="✅", url_path="ground-truth"),
        st.Page(studio.page_model, title="Data & Model", icon="🧠", url_path="model"),
    ]
    st.navigation(pages).run()
