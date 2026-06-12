"""
app.py — Streamlit Deepfake Audio Detector Web App  (improved UI)

Run:
    streamlit run app.py
"""

import json
import os
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered",
)

MODEL_DIR     = os.getenv("MODEL_DIR", "./outputs")
ENSEMBLE_PATH = os.path.join(MODEL_DIR, "ensemble_softvote.keras")
META_PATH     = os.path.join(MODEL_DIR, "ensemble_meta.json")


# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header   { visibility: hidden; }
    .stApp                      { background: #F2F4F9; }
    .block-container            { padding-top: 1rem !important; padding-bottom: 1.5rem !important; max-width: 760px; }

    /* kill streamlit's default gaps */
    .element-container          { margin-bottom: 0 !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.35rem !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #181B2A !important;
        border-right: 1px solid #252840 !important;
    }
    /* all text inside sidebar → white/light */
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p {
        color: #B8C1DC !important;
        font-size: 0.84rem;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #FFFFFF !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stSidebar"] strong { color: #E2E8F8 !important; font-weight: 600 !important; }
    /* sidebar table */
    [data-testid="stSidebar"] table { width: 100%; border-collapse: collapse; margin-top: 0.25rem; }
    [data-testid="stSidebar"] th {
        color: #6B7BA0 !important; font-size: 0.72rem !important;
        padding: 5px 8px !important; border-bottom: 1px solid #252840 !important;
        text-transform: uppercase; letter-spacing: 0.06em;
        background: rgba(255,255,255,0.03) !important;
    }
    [data-testid="stSidebar"] td {
        color: #B8C1DC !important; font-size: 0.8rem !important;
        padding: 5px 8px !important; border-bottom: 1px solid #1E2135 !important;
    }
    [data-testid="stSidebar"] tr:last-child td { border-bottom: none !important; font-weight: 700 !important; color: #E2E8F8 !important; }
    /* sidebar divider */
    [data-testid="stSidebar"] hr { border-color: #2D3148 !important; margin: 0.75rem 0 !important; }

    /* ── Slider: label ── */
    [data-testid="stSidebar"] [data-testid="stSlider"] label {
        color: #9AA3C0 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    /* ── Slider: current value bubble text ── */
    [data-testid="stSidebar"] [data-testid="stSlider"] p,
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMax"] {
        color: #818CF8 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.78rem !important;
    }
    /* ── Slider: full track rail (unselected portion) ── */
    [data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSlider"] ~ div,
    [data-testid="stSidebar"] div[data-baseweb="slider"] > div:first-child {
        background: #2D3148 !important;
        border-radius: 4px !important;
    }
    /* ── Slider: filled portion ── */
    [data-testid="stSidebar"] div[data-baseweb="slider"] div[role="progressbar"],
    [data-testid="stSidebar"] div[data-baseweb="slider"] div[data-testid] {
        background: #6366F1 !important;
    }
    /* ── Slider: thumb ── */
    [data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] {
        background: #FFFFFF !important;
        border: 2px solid #818CF8 !important;
        box-shadow: 0 0 0 3px rgba(129,140,248,0.2) !important;
        width: 14px !important; height: 14px !important;
        border-radius: 50% !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"]:hover {
        box-shadow: 0 0 0 5px rgba(129,140,248,0.25) !important;
    }
    /* ── Slider: track container background ── */
    [data-testid="stSidebar"] div[data-baseweb="slider"] > div {
        background: #252840 !important;
    }

    /* ── Checkbox ── */
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label { color: #C8D0E8 !important; }
    [data-testid="stSidebar"] [data-testid="stCheckbox"] span[data-baseweb="checkbox"] div {
        border-color: #4A5180 !important;
        background: transparent !important;
        border-radius: 4px !important;
    }

    /* ── Divider (main area) ── */
    hr { border: none; border-top: 1px solid #DDE4F0 !important; margin: 0.6rem 0 !important; }

    /* ── App header ── */
    .app-header { display: flex; flex-direction: column; gap: 2px; margin-bottom: 0.15rem; }
    .app-header .eyebrow {
        font-family: 'Space Mono', monospace;
        font-size: 0.67rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #5046E5;
    }
    .app-header h1 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #0D1117 !important;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 0 !important; padding: 0 !important;
    }
    .app-header .subtitle { font-size: 0.83rem; color: #6B7FA8; margin-top: 1px; }

    /* ── Model tag pills ── */
    .model-tag-row { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.3rem; margin-bottom: 0.15rem; }
    .model-tag {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        padding: 0.18rem 0.55rem;
        border-radius: 6px;
        background: #ECEEFF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1.5px dashed #C5CDE8;
        border-radius: 10px;
        padding: 0.4rem 0.9rem;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #818CF8;
        box-shadow: 0 0 0 3px rgba(129,140,248,0.08);
    }
    [data-testid="stFileUploader"] label {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important; color: #1A2035 !important;
    }

    /* ── Info / warning / error boxes ── */
    [data-testid="stAlert"] { border-radius: 9px !important; font-family: 'Inter', sans-serif !important; }
    div[data-testid="stAlert"][kind="info"] {
        background: #F0F2FF !important;
        border: 1px solid #C7D2FE !important;
        color: #3730A3 !important;
    }

    /* ── Audio ── */
    audio { width: 100%; border-radius: 8px; accent-color: #5046E5; }

    /* ── Verdict card ── */
    .verdict-card {
        position: relative;
        overflow: hidden;
        border-radius: 14px;
        padding: 1.5rem 2rem 1.4rem;
        margin: 0.4rem 0;
        text-align: center;
    }
    .verdict-card.genuine { background: linear-gradient(135deg,#ECFDF5,#DCFCE7); border: 1.5px solid #86EFAC; }
    .verdict-card.fake    { background: linear-gradient(135deg,#FFF1F2,#FFE4E6); border: 1.5px solid #FCA5A5; }

    .verdict-card::before {
        content: ''; position: absolute; top: -2px; left: 0;
        width: 100%; height: 2px; opacity: 0.5;
        animation: scanline 2.2s ease-in-out 0.2s 1 forwards;
    }
    .verdict-card.genuine::before { background: #16A34A; }
    .verdict-card.fake::before    { background: #DC2626; }
    @keyframes scanline { 0%{top:-2px;opacity:.6} 100%{top:100%;opacity:0} }

    .verdict-label {
        font-family: 'Space Mono', monospace;
        font-size: clamp(2rem,6.5vw,3.4rem);
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .verdict-label.genuine { color: #15803D; }
    .verdict-label.fake    { color: #B91C1C; }

    .verdict-descriptor {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }
    .verdict-descriptor.genuine { color: #16A34A; }
    .verdict-descriptor.fake    { color: #DC2626; }

    .verdict-metrics { display: flex; justify-content: center; gap: 2.2rem; flex-wrap: wrap; }
    .verdict-metric  { display: flex; flex-direction: column; align-items: center; gap: 1px; }
    .verdict-metric .val {
        font-family: 'Space Mono', monospace;
        font-size: 1.4rem; font-weight: 700; color: #0D1117;
    }
    .verdict-metric .lbl {
        font-size: 0.66rem; letter-spacing: 0.1em;
        text-transform: uppercase; color: #6B7FA8;
    }

    /* ── Prob bars ── */
    .prob-section {
        background: #FFFFFF;
        border: 1px solid #DDE4F0;
        border-radius: 10px;
        padding: 1rem 1.4rem 1rem;
        margin-top: 0.4rem;
    }
    .prob-section h4 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
        color: #6B7FA8; margin: 0 0 0.75rem 0;
    }
    .prob-row          { margin-bottom: 0.75rem; }
    .prob-row:last-child { margin-bottom: 0; }
    .prob-row-header   { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.3rem; }
    .prob-row-header .prob-name { font-family: 'Plus Jakarta Sans',sans-serif; font-size: 0.85rem; font-weight: 600; color: #1A2035; }
    .prob-row-header .prob-val  { font-family: 'Space Mono',monospace; font-size: 0.8rem; color: #4A5573; }
    .prob-track { height: 7px; background: #EEF1F8; border-radius: 99px; overflow: hidden; }
    .prob-fill  { height: 100%; border-radius: 99px; }
    .prob-fill.genuine { background: linear-gradient(90deg,#34D399,#16A34A); }
    .prob-fill.fake    { background: linear-gradient(90deg,#FB7185,#DC2626); }

    /* ── Info pill ── */
    .info-pill {
        display: inline-flex; align-items: center; gap: 0.35rem;
        background: #ECEEFF; color: #4338CA;
        font-family: 'Space Mono',monospace; font-size: 0.68rem;
        padding: 0.25rem 0.7rem; border-radius: 99px; margin-top: 0.5rem;
    }

    /* ── Suhani credit ── */
    .suhani-credit {
        position: fixed;
        right: 16px; bottom: 14px;
        background: #181B2A;
        padding: 5px 14px 5px 10px;
        border-radius: 20px;
        display: flex; align-items: center; gap: 7px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.22), 0 0 0 1px rgba(129,140,248,0.15);
        z-index: 9999;
        border: 1px solid #2D3148;
    }
    .suhani-credit .dot {
        width: 5px; height: 5px;
        background: #818CF8;
        border-radius: 50%;
        box-shadow: 0 0 5px rgba(129,140,248,0.75);
        flex-shrink: 0;
    }
    .suhani-credit span {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: #7A85A8 !important;
        letter-spacing: 0.03em;
        white-space: nowrap;
    }
    .suhani-credit strong { color: #A5B0FF !important; font-weight: 700 !important; }

    /* ── Download button ── */
    [data-testid="stDownloadButton"] button {
        background: #FFFFFF !important; color: #1A2035 !important;
        border: 1.5px solid #DDE4F0 !important; border-radius: 8px !important;
        font-family: 'Plus Jakarta Sans',sans-serif !important;
        font-weight: 600 !important; font-size: 0.84rem !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        border-color: #5046E5 !important;
        box-shadow: 0 0 0 3px rgba(80,70,229,0.1) !important;
    }

    /* ── Caption ── */
    .stCaption, small { color: #8896B0 !important; font-size: 0.75rem !important; }

    /* ── Info/warning/error boxes ── */
    [data-testid="stAlert"] { border-radius: 9px !important; }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Custom layer registration
# ═══════════════════════════════════════════════════════════════════════════
def _register_custom_layers():
    import keras as ks
    @ks.saving.register_keras_serializable()
    class SumPool1D(ks.layers.Layer):
        def call(self, x):      return ks.ops.sum(x, axis=1)
        def get_config(self):   return super().get_config()
    return {"SumPool1D": SumPool1D}


@st.cache_resource(show_spinner="Loading ensemble model …")
def load_model_and_meta():
    from tensorflow.keras.models import load_model
    model = load_model(ENSEMBLE_PATH, custom_objects=_register_custom_layers())
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════
def load_audio(path, sr=16_000, n_samples=48_000):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    else:
        start = (len(y) - n_samples) // 2
        y = y[start: start + n_samples]
    return y.astype(np.float32)


def extract_mfcc_sequence(y, sr=16_000, n_mfcc=40, n_fft=512, hop_length=160, T=300):
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)
    if mfcc.shape[1] < T:
        mfcc = np.pad(mfcc, ((0,0),(0, T - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :T]
    return mfcc.T.astype(np.float32)


def run_prediction(audio_bytes, filename, model, meta):
    cfg       = meta.get("cfg", {})
    threshold = float(meta.get("threshold", 0.5))
    suffix    = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes); tmp_path = tmp.name
    try:
        y   = load_audio(tmp_path, sr=cfg.get("sr",16_000), n_samples=cfg.get("n_samples",48_000))
        seq = extract_mfcc_sequence(y, sr=cfg.get("sr",16_000), n_mfcc=cfg.get("n_mfcc",40),
                                    n_fft=cfg.get("n_fft",512), hop_length=cfg.get("hop_length",160),
                                    T=cfg.get("T",300))
        p_fake = float(model.predict(seq[np.newaxis,...], verbose=0)[0,0])
    finally:
        os.unlink(tmp_path)
    p_real  = 1.0 - p_fake
    is_fake = p_fake >= threshold
    conf    = p_fake if is_fake else p_real
    return {"label": "Deepfake (AI-Generated)" if is_fake else "Genuine (Human)",
            "is_fake": is_fake, "confidence": conf,
            "p_fake": p_fake, "p_real": p_real, "threshold": threshold}


# ═══════════════════════════════════════════════════════════════════════════
# Waveform plot
# ═══════════════════════════════════════════════════════════════════════════
def plot_waveform(audio_bytes, filename):
    import librosa, librosa.display, matplotlib.pyplot as plt, matplotlib as mpl
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes); tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=16_000, mono=True, duration=3.0)
    finally:
        os.unlink(tmp_path)

    mpl.rcParams.update({"font.family":"monospace","axes.spines.top":False,
                          "axes.spines.right":False,"axes.labelcolor":"#6B7FA8",
                          "xtick.color":"#6B7FA8","ytick.color":"#6B7FA8",
                          "xtick.labelsize":8,"ytick.labelsize":8})
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.6))
    fig.patch.set_facecolor("#FFFFFF")
    for ax in axes:
        ax.set_facecolor("#F8F9FF")
        ax.spines["left"].set_color("#DDE4F0")
        ax.spines["bottom"].set_color("#DDE4F0")

    t = np.linspace(0, len(y)/sr, len(y))
    axes[0].plot(t, y, lw=0.6, color="#5046E5", alpha=0.9)
    axes[0].fill_between(t, y, alpha=0.07, color="#5046E5")
    axes[0].set(xlabel="Time (s)", ylabel="Amplitude", title="Waveform")
    axes[0].title.set(color="#1A2035", fontsize=10, fontweight="bold")

    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img    = librosa.display.specshow(mel_db, sr=sr, hop_length=160,
                                      x_axis="time", y_axis="mel", ax=axes[1], cmap="RdPu")
    axes[1].set(title="Mel Spectrogram")
    axes[1].title.set(color="#1A2035", fontsize=10, fontweight="bold")
    cb = fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    cb.ax.tick_params(labelcolor="#6B7FA8", labelsize=7)
    fig.tight_layout(pad=1.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# HTML components
# ═══════════════════════════════════════════════════════════════════════════
def render_verdict_card(result):
    kind  = "fake" if result["is_fake"] else "genuine"
    icon  = "🔴" if result["is_fake"] else "🟢"
    lbl   = "DEEPFAKE"   if result["is_fake"] else "GENUINE"
    desc  = "AI-Generated Audio" if result["is_fake"] else "Human Speech"
    conf  = f"{result['confidence']*100:.1f}%"
    pfake = f"{result['p_fake']:.4f}"
    preal = f"{result['p_real']:.4f}"

    st.markdown(f"""
    <div class="verdict-card {kind}">
        <div class="verdict-label {kind}">{icon}&nbsp;{lbl}</div>
        <div class="verdict-descriptor {kind}">{desc}</div>
        <div class="verdict-metrics">
            <div class="verdict-metric"><span class="val">{conf}</span><span class="lbl">Confidence</span></div>
            <div class="verdict-metric"><span class="val">{pfake}</span><span class="lbl">P(Fake)</span></div>
            <div class="verdict-metric"><span class="val">{preal}</span><span class="lbl">P(Real)</span></div>
        </div>
    </div>""", unsafe_allow_html=True)


def render_prob_bars(result):
    rw = int(result["p_real"] * 100)
    fw = int(result["p_fake"] * 100)
    st.markdown(f"""
    <div class="prob-section">
        <h4>Model Output Probabilities</h4>
        <div class="prob-row">
            <div class="prob-row-header">
                <span class="prob-name">Genuine (Human)</span>
                <span class="prob-val">{result['p_real']*100:.1f}%</span>
            </div>
            <div class="prob-track"><div class="prob-fill genuine" style="width:{rw}%"></div></div>
        </div>
        <div class="prob-row">
            <div class="prob-row-header">
                <span class="prob-name">Deepfake (AI-Generated)</span>
                <span class="prob-val">{result['p_fake']*100:.1f}%</span>
            </div>
            <div class="prob-track"><div class="prob-fill fake" style="width:{fw}%"></div></div>
        </div>
    </div>
    <div><span class="info-pill">⚙&nbsp;threshold&nbsp;{result['threshold']:.3f}</span></div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Main UI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    inject_css()

    # ── Suhani credit (fixed bottom-right) ────────────────────────────────
    st.markdown("""
    <div class="suhani-credit">
        <span class="dot"></span>
        <span>made by <strong>suhani</strong></span>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("About")
        st.markdown(
            "Classifies audio as **Genuine (Human)** or **Deepfake (AI-Generated)**.\n\n"
            "**Architecture** — soft-voting ensemble\n\n"
            "| Model | Accuracy |\n"
            "|---|---|\n"
            "| 1D-CNN | 82.6% |\n"
            "| BiLSTM-Attention | 80.5% |\n"
            "| CNN-BiLSTM | 89.2% |\n"
            "| **Ensemble** | **91.1%** |\n\n"
            "**Features:** MFCC (40 coeff.) · 16 kHz\n\n"
            "**Dataset:** Fake-or-Real (FoR) for-norm"
        )
        st.divider()
        custom_thr = st.slider(
            "Decision threshold",
            min_value=0.1, max_value=0.9, value=0.535, step=0.005,
            help="Default 0.535 is optimised on the validation set. "
                 "Lower → more sensitive to fakes. Higher → more conservative.",
        )
        show_viz = st.checkbox("Show waveform & spectrogram", value=True)

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <span class="eyebrow">Audio Forensics Tool</span>
        <h1>Deepfake Audio Detector</h1>
        <p class="subtitle">Ensemble · 1D-CNN + BiLSTM-Attention + CNN-BiLSTM &nbsp;·&nbsp; Accuracy 91.1% &nbsp;·&nbsp; EER 7.8% &nbsp;·&nbsp; F1 91.6%</p>
    </div>
    <div class="model-tag-row">
        <span class="model-tag">MFCC-40</span>
        <span class="model-tag">16 kHz</span>
        <span class="model-tag">FoR Dataset</span>
        <span class="model-tag">Soft-Vote Ensemble</span>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Model check ────────────────────────────────────────────────────────
    if not os.path.isfile(ENSEMBLE_PATH):
        st.error(
            f"**Ensemble model not found:** `{ENSEMBLE_PATH}`\n\n"
            "Train the model in the notebook first, then copy `outputs/` here."
        )
        st.stop()

    model, meta = load_model_and_meta()
    meta["threshold"] = custom_thr

    # ── Upload ─────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop an audio file here",
        type=["wav","flac","mp3"],
        help="WAV, FLAC, or MP3 · recommended 3+ seconds of speech",
    )

    if uploaded is None:
        st.info("Upload a **.wav**, **.flac**, or **.mp3** file to get started.")
        return

    audio_bytes = uploaded.read()
    st.audio(audio_bytes, format=f"audio/{uploaded.name.split('.')[-1]}")
    st.caption(f"📄 {uploaded.name}")

    # ── Waveform ───────────────────────────────────────────────────────────
    if show_viz:
        with st.spinner("Rendering signal …"):
            try:   plot_waveform(audio_bytes, uploaded.name)
            except Exception as e: st.warning(f"Could not render waveform: {e}")

    # ── Predict ────────────────────────────────────────────────────────────
    with st.spinner("Analysing audio …"):
        try:   result = run_prediction(audio_bytes, uploaded.name, model, meta)
        except Exception as e:
            st.error(f"Prediction failed: {e}"); return

    render_verdict_card(result)
    render_prob_bars(result)

    st.divider()

    payload = {"file": uploaded.name, "label": result["label"],
               "confidence": round(result["confidence"],4),
               "p_fake": round(result["p_fake"],4), "p_real": round(result["p_real"],4),
               "threshold": result["threshold"]}
    st.download_button(
        label="⬇ Download result as JSON",
        data=json.dumps(payload, indent=2),
        file_name=f"deepfake_result_{Path(uploaded.name).stem}.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()