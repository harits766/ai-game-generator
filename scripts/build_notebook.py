"""Bangun notebooks/capstone_analysis.ipynb (lalu eksekusi dengan nbconvert).

python scripts/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/capstone_analysis.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# AI Game Generator: Analisis Data, Modeling, dan GenAI
**Capstone Project Data Science & Generative AI (Celerates)**

**Masalah.** Acara kumpul (kajian, kelas, arisan, sosialisasi desa) sering berakhir dengan semua orang sibuk
main HP sendiri-sendiri. Game party yang ada kaku: pilihannya itu-itu saja (mis. cuma satu jenis game),
harus gonta-ganti aplikasi, dan tidak bisa menyesuaikan materi acara.

**Solusi.** Aplikasi yang (1) membaca tren minat publik terhadap tiap jenis game dari data **real**
(Wikipedia pageviews + Google Trends Indonesia), (2) memakai data **real** BoardGameGeek untuk menilai daya tarik
game baru yang belum punya riwayat (cold start), (3) mencampur keduanya dengan log permintaan di aplikasi yang
**mulai dari nol** dan terus bertambah, lalu (4) memakai agent LangChain untuk membuat game dari dokumen acara dan
langsung memberi link room host + peserta.

| Sumber | Status | Isi |
|---|---|---|
| Wikipedia Pageviews API | real | view mingguan artikel yang mewakili tiap jenis game (21 seri) |
| Google Trends Indonesia | real | indeks minat pencarian 0-100 per keyword game (15 seri) |
| BoardGameGeek (TidyTuesday 2022) | real | 4.911 game party: profil + jumlah rating |
| Log request in-app | real, mulai 0 | belum ada saat notebook ini dibuat -> bagian 7 memakai **SIMULASI berlabel** |
| TikTok / X | tidak dipakai | tidak ada API gratis yang stabil |
""")
code("""
import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize': (10, 4), 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .3})
from core.catalog import GAMES, GAME_BY_ID
from core.real_sources import GAME_SOURCES, CONTEXT_WIKI
NAMES = {g['id']: g['name'] for g in GAMES} | {f'ctx:{k}': f'[konteks] {k}' for k in CONTEXT_WIKI}
REAL = ROOT / 'data' / 'real'
""")
md("## 1. Pemahaman Data\n### 1.1 Sinyal tren real (Wikipedia + Google Trends)")
code("""
wiki = pd.read_csv(REAL / 'wiki_weekly.csv', parse_dates=['week'])
gt = pd.read_csv(REAL / 'gtrends_weekly.csv', parse_dates=['week'])
log = json.loads((REAL / 'fetch_log.json').read_text())
print('wiki   :', wiki.shape, '| seri:', wiki.game_id.nunique(), '| minggu:', wiki.week.min().date(), '->', wiki.week.max().date())
print('gtrends:', gt.shape, '| seri:', gt.game_id.nunique())
print('diambil:', log['run_at'])
pd.DataFrame([{'game': NAMES.get(g, g), 'artikel Wikipedia': ', '.join(s['wiki']), 'keyword Google Trends': s['gtrends']}
              for g, s in GAME_SOURCES.items()])
""")
code("""
desc = wiki.groupby('game_id')['wiki_views'].agg(['count', 'mean', 'median', 'min', 'max']).round(0)
desc.index = desc.index.map(lambda g: NAMES.get(g, g)); desc.sort_values('median', ascending=False)
""")
code("gt.groupby('keyword')['google_trends'].describe()[['count','mean','50%','min','max']].round(1)")
md("### 1.2 BoardGameGeek (data real game party)")
code("""
from core.appeal import load_bgg
bgg = load_bgg()
print(bgg.shape); bgg[['name','yearpublished','minplayers','maxplayers','playingtime','minage','users_rated','average']].head()
""")
code("""
info = pd.DataFrame({'missing': bgg.isna().sum(), 'unique': bgg.drop(columns=['categories','mechanics']).nunique()})
print(info.T); bgg[['minplayers','maxplayers','playingtime','minage','users_rated','average','bayes_average']].describe().T.round(1)
""")
md("""
## 2. Preprocessing
- **Tren:** data harian Wikipedia diagregasi ke minggu (Senin), hanya minggu lengkap (7 hari). Beberapa artikel
  dijumlahkan bila satu jenis game diwakili >1 artikel (mis. *Fact-checking* + *Hoaks*). Nilai di-log
  (`log1p`) karena skala antar artikel sangat beda (puluhan sampai jutaan view). Google Trends digabung per minggu;
  yang kosong diberi nilai -1 + flag `has_gt` (bukan dibuang).
- **BGG:** kolom kategori/mekanik (string list) di-parse; hanya game terbit 2000-2021 (game terlalu lama atau terlalu
  baru bias jumlah rating); outlier `maxplayers` (999) dan `playingtime` (10.000 menit) di-clip.
""")
code("""
from core.trend_model import build_panel, FEATURES, TARGET
panel = build_panel()
print('panel tren:', panel.shape, '| fitur:', len(FEATURES)); print('cakupan Google Trends:', panel.has_gt.mean().round(3))
panel[['game_id','week','wiki_views','lv_t','roll12','mom12','google_trends', TARGET]].tail()
""")
md("## 3. Exploratory Data Analysis")
code("""
fig, ax = plt.subplots(figsize=(11, 5))
for gid in ['kuis_kilat','survival_kuis','tebak_kata_kunci','voting_meme','satu_frekuensi','adu_receh']:
    d = wiki[wiki.game_id == gid]; ax.plot(d.week, d.wiki_views, label=NAMES[gid], lw=1.6)
ax.set_yscale('log'); ax.set_ylabel('views / minggu (skala log)'); ax.set_title('Wikipedia pageviews mingguan (real)'); ax.legend(ncol=3, fontsize=8); plt.show()
""")
md("**Insight:** skala minat sangat berbeda antar jenis game, dan beberapa punya lonjakan tajam (mis. *Squid Game* saat musim baru tayang). Karena itu ranking memakai **momentum** (dibanding rata-rata 12 minggu sendiri), bukan angka mentah.")
code("""
pv = gt.pivot_table(index='keyword', columns=gt.week.dt.to_period('M').astype(str), values='google_trends')
fig, ax = plt.subplots(figsize=(13, 5)); im = ax.imshow(pv.values, aspect='auto', cmap='Purples')
ax.set_yticks(range(len(pv)), pv.index); ax.set_xticks(range(0, pv.shape[1], 2), pv.columns[::2], rotation=45, ha='right')
ax.grid(False); plt.colorbar(im, label='Google Trends (0-100)'); ax.set_title('Minat pencarian Indonesia per bulan (real)'); plt.show()
""")
code("""
m = wiki.merge(gt, on=['week','game_id'])
corr = m.groupby('game_id').apply(lambda d: np.log1p(d.wiki_views).corr(d.google_trends)).rename(NAMES).sort_values()
corr.plot.barh(color='#6D28D9', title='Korelasi Wikipedia (log) vs Google Trends ID per game', figsize=(9,5)); plt.show()
print('median korelasi:', corr.median().round(2))
""")
md("**Insight:** Wikipedia (global, mayoritas bahasa Inggris) dan Google Trends Indonesia tidak selalu searah. Keduanya dipakai bersama: Wikipedia untuk skala & prediksi, Google Trends untuk konteks Indonesia.")
code("""
d = panel.assign(delta=panel[TARGET] - panel.lv_t).dropna(subset=['delta'])
ac = d.groupby('game_id').apply(lambda x: x.delta.autocorr(1)).median()
print('median autokorelasi perubahan mingguan (lag 1):', round(ac, 3))
d.delta.plot.hist(bins=60, color='#DB2777', title='Distribusi perubahan log views mingguan'); plt.show()
""")
md("**Insight penting:** perubahan mingguan cenderung berbalik (autokorelasi negatif = *mean reversion*) dan sebagian besar kecil. Data seperti ini mirip *random walk*, sehingga baseline naive sangat kuat. Ini alasan model memprediksi **perubahan (delta)**, bukan level.")
code("""
fig, axs = plt.subplots(1, 3, figsize=(16, 4))
b = bgg[bgg.yearpublished.between(2000, 2021)]
np.log10(b.users_rated + 1).plot.hist(bins=40, ax=axs[0], color='#6D28D9', title='log10 jumlah rating (BGG)')
b.playingtime.clip(0, 180).plot.hist(bins=36, ax=axs[1], color='#D97706', title='Durasi main (menit, clip 180)')
b.maxplayers.clip(0, 20).value_counts().sort_index().plot.bar(ax=axs[2], color='#059669', title='Maks pemain (clip 20)', rot=0)
plt.tight_layout(); plt.show()
""")
code("""
thr = b.users_rated.quantile(.75); b = b.assign(populer=b.users_rated >= thr)
ex = b.explode('categories'); rate = ex.groupby('categories').populer.agg(['mean','size']).query('size >= 80').sort_values('mean')
rate['mean'].plot.barh(color='#2563EB', figsize=(9, 7), title=f'Peluang populer per kategori (populer = rating >= {thr:.0f})'); plt.axvline(.25, color='k', ls='--', lw=1); plt.show()
""")
code("""
ex = b.explode('mechanics'); rate = ex.groupby('mechanics').populer.agg(['mean','size']).query('size >= 80').sort_values('mean')
rate['mean'].plot.barh(color='#DB2777', figsize=(9, 8), title='Peluang populer per mekanik'); plt.axvline(.25, color='k', ls='--', lw=1); plt.show()
""")
md("**Insight:** mekanik interaksi sosial (voting, tim, deduksi) dan durasi pendek lebih sering populer di kategori party. Informasi ini yang dipakai model daya tarik untuk menilai game baru.")
md("""
## 4. Model 1: prediksi tren real (>= 3 model)
Target: perubahan `log(1+views)` minggu depan. Split waktu (20 minggu terakhir = test).
Metrik: MAE log (~error relatif), MAPE views, akurasi arah naik/turun, Precision@10 ranking momentum.
""")
code("""
from scripts.train_model import main as train_trend
rep = train_trend()
res = pd.DataFrame(rep['results']).set_index('model'); res
""")
code("""
fig, axs = plt.subplots(1, 3, figsize=(17, 4))
res['MAE_log'].sort_values().plot.barh(ax=axs[0], color='#6D28D9', title='MAE log (kecil = baik)')
res['Akurasi_arah'].sort_values().plot.barh(ax=axs[1], color='#059669', title='Akurasi arah naik/turun', xlim=(0, .7)); axs[1].axvline(.5, color='k', ls='--', lw=1)
res['Precision@10_momentum'].sort_values().plot.barh(ax=axs[2], color='#DB2777', title='Precision@10 momentum', xlim=(.6, 1))
plt.tight_layout(); plt.show(); print('Model terpilih:', rep['best_model'])
""")
md("""
**Interpretasi jujur.** Baseline naive (minggu depan = minggu ini) sudah sangat kuat di data web. Model terpilih
mengalahkan baseline di MAE dan menebak arah naik/turun > 50% (baseline naive tidak bisa menebak arah sama sekali),
sehingga berguna untuk **momentum**, yaitu jenis game mana yang sedang naik daun.
""")
code("""
import joblib
from core.config import MODEL_PATH
bundle = joblib.load(MODEL_PATH)
data = panel.dropna(subset=[TARGET]); weeks = sorted(data.week.unique()); test = data[data.week >= weeks[-20]].copy()
test['pred'] = test.lv_t + bundle['model'].predict(test[FEATURES])
fig, axs = plt.subplots(1, 3, figsize=(16, 4))
for ax, gid in zip(axs, ['kuis_kilat', 'survival_kuis', 'tebak_kata_kunci']):
    d = test[test.game_id == gid]
    ax.plot(d.week, np.expm1(d[TARGET]), color='#1F1534', label='aktual'); ax.plot(d.week, np.expm1(d.pred), '--', color='#DB2777', label='prediksi')
    ax.set_title(NAMES[gid]); ax.tick_params(axis='x', rotation=30)
axs[0].legend(); plt.tight_layout(); plt.show()
pd.Series(rep['feature_importance']).head(10).sort_values().plot.barh(color='#DB2777', title='Feature importance model tren'); plt.show()
""")
md("""
## 5. Model 2: daya tarik game (cold start, data real BGG, >= 3 model)
Klasifikasi: apakah game party dengan profil tertentu (pemain, durasi, usia, kategori, mekanik) masuk 25% teratas
jumlah rating. Evaluasi: Stratified 5-fold CV + holdout.
""")
code("""
from scripts.train_appeal_model import main as train_appeal
ap = train_appeal()
pd.DataFrame(ap['results']).set_index('model')
""")
code("""
pd.Series(ap['feature_importance']).head(12).sort_values().plot.barh(color='#2563EB', title='Feature importance model daya tarik'); plt.show()
from core.appeal import appeal_score
from core.catalog import MECHANICS
pd.Series({m: appeal_score({'mechanic': m, 'max_players': 20, 'rounds': 8, 'seconds_per_round': 20}) for m in MECHANICS}).round(3).rename('P(populer)')
""")
md("""
## 6. Skor ranking library
`skor = (1 - w) x [0,65 x tren_real + 0,35 x daya_tarik] + w x request_in_app`

- `tren_real` = 0,40 x momentum prediksi (Wikipedia) + 0,35 x momentum Google Trends ID + 0,25 x level views (semua dalam persentil)
- `daya_tarik` = probabilitas populer dari model BGG (game custom hanya memakai ini + request)
- `w = n / (n + 50)`, maks 0,7, dengan `n` = request in-app 28 hari terakhir. **Mulai dari 0.**
""")
code("""
import tempfile, core.library as L, core.storage as S
tmp = Path(tempfile.mkdtemp())
L.RUNTIME_DIR, L.LIBRARY_STATE, L.RUNTIME_REQUESTS = tmp, tmp/'state.json', tmp/'req.csv'
S.RUNTIME_REQUESTS = tmp/'req.csv'; L._pred_cache.clear()
sc = L.compute_scores()
sc[['name','views_t','momentum','gt_mom','appeal','trend_part','w_req','score']].round(3).head(15)
""")
md("""
## 7. SIMULASI (bukan data real): bagaimana request in-app menggeser ranking
Log request in-app masih nol saat notebook ini dibuat. Bagian ini **mensimulasikan** 0 sampai 300 request
(mis. komunitas yang sering memilih *Siapa Paling*) untuk menunjukkan bobot `w` naik dan ranking bergeser.
""")
code("""
rows = []
target = 'siapa_paling'
rng = np.random.default_rng(0); others = [g['id'] for g in GAMES if g['id'] != target]
done = 0
for n in [0, 10, 25, 50, 100, 200, 300]:
    while done < n:
        L.log_request(target if rng.random() < .4 else rng.choice(others), {'players': 10}); done += 1
    s = L.compute_scores()
    rows.append({'request_simulasi': n, 'w_request': s.w_req.iloc[0], 'ranking_siapa_paling': int(s.index[s.game_id == target][0]) + 1})
sim = pd.DataFrame(rows); print(sim.to_string(index=False))
ax = sim.plot(x='request_simulasi', y='ranking_siapa_paling', marker='o', color='#DB2777', legend=False, title='SIMULASI: ranking game yang sering diminta')
ax.invert_yaxis(); ax.set_ylabel('ranking (1 = teratas)'); plt.show()
""")
md("""
## 8. Validasi ground truth: apakah Top-10 masih sama?
Setiap update (otomatis tiap Senin via GitHub Actions), ground truth menghitung Top-10 baru dan membandingkan dengan
versi sebelumnya: overlap, Kendall tau, game masuk/keluar. **STABIL** jika overlap >= 8/10.
Di bawah ini replay mingguan selama periode test dengan data real (komponen tren saja, tanpa request in-app).
""")
code("""
from core.ground_truth import compare
cat = test[test.game_id.isin(GAME_BY_ID)].copy(); cat['mom'] = cat.pred - cat.roll12
snaps, prev = [], []
for w, d in cat.groupby('week'):
    d = d.assign(s=d.mom.rank(pct=True) * .6 + d.lv_t.rank(pct=True) * .4)
    top = list(d.nlargest(10, 's').game_id); c = compare(prev, top)
    snaps.append({'week': w, 'overlap': c['overlap'], 'verdict': c['verdict'], 'masuk': ','.join(c['masuk']) if prev else ''})
    prev = top
gt_sim = pd.DataFrame(snaps); print(gt_sim.verdict.value_counts().to_dict())
ax = gt_sim.set_index('week').overlap.plot(marker='o', color='#6D28D9', ylim=(0, 10.5), title='Overlap Top-10 dengan minggu sebelumnya (data real)')
ax.axhline(8, color='#E11D48', ls='--', lw=1); plt.show(); gt_sim.tail(8)
""")
md("""
## 9. Generative AI: agent LangChain
**Komponen (semua lewat `core/llm.py`, satu-satunya tempat API key dipakai):**
1. **Agent Game Master** (`langchain.agents.create_agent`, ReAct di atas LangGraph) dengan 7 tools:
   `get_trends`, `recommend_games`, `read_document`, `design_custom_game`, `validate_custom_game`,
   `register_custom_game`, `create_room`.
2. **Chain Desainer**: ide bebas -> `GameSpec` (Pydantic) -> validasi aturan kunci.
3. **Chain Konten**: potongan dokumen (retrieval TF-IDF, maks 4 potongan) -> `RoundsOut` (Pydantic).

**Pengaman:** ModelFallback (Gemini -> OpenAI), maks 6 panggilan LLM per pesan, `create_room` maks 1x per pesan,
total tool call maks 8, output token maks 1.500, temperature rendah, dan semua angka/link harus berasal dari tool.
Tanpa API key, planner offline memanggil tools yang sama.
""")
code("""
from core.documents import parse_bytes
from core.generator import generate_rounds
doc = parse_bytes('materi_pemilahan_sampah.md', (ROOT/'samples/materi_pemilahan_sampah.md').read_bytes())
for gid in ['kuis_kilat', 'fakta_hoaks', 'tebak_angka']:
    rounds, meta = generate_rounds(doc, GAME_BY_ID[gid], 2, 'sampah', use_llm=False)
    print(f"[{NAMES[gid]}] sumber={meta['provider']}")
    for r in rounds: print('   ', json.dumps(r, ensure_ascii=False)[:170])
""")
code("""
# demo loop agent LangChain dengan model palsu (tanpa API key) untuk memperlihatkan alur tool-calling
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from core.agent import AgentContext, build_tools, SYSTEM
from core.lc_agent import run_agent
class Fake(GenericFakeChatModel):
    def bind_tools(self, tools, **kw): return self
tmp2 = Path(tempfile.mkdtemp())   # state bersih: buang request simulasi bagian 7
L.RUNTIME_DIR, L.LIBRARY_STATE, L.RUNTIME_REQUESTS = tmp2, tmp2/'state.json', tmp2/'req.csv'
S.RUNTIME_REQUESTS = tmp2/'req.csv'; L._pred_cache.clear()
script = iter([AIMessage('', tool_calls=[{'name': 'recommend_games', 'args': {'players': 20, 'duration_min': 30, 'event_type': 'sosialisasi desa'}, 'id': '1'}]),
               AIMessage('', tool_calls=[{'name': 'create_room', 'args': {'game_id': 'fakta_hoaks', 'players': 20}, 'id': '2'}]),
               AIMessage('Room siap, ini link host & peserta.')])
ctx = AgentContext(base_url='https://ai-game-generator.streamlit.app', doc=doc, use_llm=False)
answer, trace = run_agent([Fake(messages=script)], build_tools(ctx), SYSTEM, [{'role': 'user', 'content': 'sosialisasi desa 20 warga 30 menit'}])
for t in trace: print(t['tool'], '->', {k: t['result'][k] for k in list(t['result'])[:3]})
print(answer)
""")
md("""
## 10. Kesimpulan
1. Data real (Wikipedia + Google Trends) menunjukkan minat mingguan mirip random walk; model tren (delta) tetap
   mengalahkan baseline dan berguna untuk mendeteksi **momentum** jenis game.
2. Data real BoardGameGeek memberi model daya tarik (ROC-AUC ~0,7) untuk **cold start** game baru/custom.
3. Log request in-app dimulai dari nol; bobotnya naik otomatis, sehingga ranking makin mencerminkan pemakaian nyata.
4. Ground truth tunggal, diperbarui tiap minggu oleh GitHub Actions, membuat keputusan library bisa diaudit (STABIL/BERUBAH).
5. Agent LangChain dengan 7 tools dan pengaman pemakaian API mengubah permintaan sehari-hari jadi room siap main.

**Saran:** tambah sumber Indonesia lain (YouTube Data API, dengan key gratis), kumpulkan rating per ronde untuk
mengukur kualitas soal, dan pindahkan state room ke database bila pemain > 30.
""")
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "capstone_analysis.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("notebook ->", out)
