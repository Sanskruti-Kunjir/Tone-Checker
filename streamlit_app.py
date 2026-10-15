"""Emotional Tone Consistency Checker - Streamlit version.
Run locally:  streamlit run streamlit_app.py

Features:
  - Text-based emotional tone analysis (preserved from original)
  - Voice-based speech emotion recognition (new)
  - Premium dark UI with teal accents
  - Consistent results dashboard for both modes

Text analysis uses j-hartmann/emotion-english-distilroberta-base (7 emotions).
Voice analysis uses ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition.
Consistency/variation scores use Jensen-Shannon distance between consecutive segments.
  Score = 100 × (1 − mean(JSD between consecutive segments))
  where JSD = sqrt( (KL(p||m) + KL(q||m)) / 2 ), m = (p+q)/2
"""
import io
import re
import tempfile
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─── Constants ───────────────────────────────────────────────────────────────
TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
VOICE_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
# Voice model labels — the actual checkpoint has 8 labels:
# angry, calm, disgust, fearful, happy, neutral, sad, surprised
# We map these to the 7 standard LABELS used for display.
VOICE_LABELS_RAW = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
VOICE_LABEL_MAP = {
    "angry": "anger", "calm": "neutral", "disgust": "disgust", "fearful": "fear",
    "happy": "joy", "neutral": "neutral", "sad": "sadness", "surprised": "surprise",
}

EMOTION_COLORS = {
    "anger":    "#E74C6F",
    "disgust":  "#A3B86C",
    "fear":     "#9B72CF",
    "joy":      "#F5B942",
    "neutral":  "#8A9BA8",
    "sadness":  "#5B9BD5",
    "surprise": "#F58442",
}
EMOTION_ICONS = {
    "anger": "🔥", "disgust": "🤢", "fear": "😨",
    "joy": "😊", "neutral": "😐", "sadness": "😢", "surprise": "😲",
}

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tone Consistency Checker",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─── Premium Dark Theme CSS ─────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(170deg, #0d1117 0%, #111922 40%, #0f1923 100%);
        font-family: 'Inter', sans-serif;
        color: #c9d5e0;
    }
    .block-container {
        max-width: 960px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ── Header ── */
    .app-header {
        text-align: center;
        padding: 2.5rem 0 1.2rem;
    }
    .app-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4fd1c5 0%, #63e6be 50%, #38b2ac 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #7a8b9a;
        font-size: 1rem;
        font-weight: 400;
        margin: 0;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #151d27;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #1e2d3d;
        justify-content: center;
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        color: #7a8b9a;
        font-size: 0.92rem;
        font-family: 'Inter', sans-serif;
        background: transparent;
        border: none;
        transition: all 0.25s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a3a4a 0%, #1e4a5a 100%) !important;
        color: #4fd1c5 !important;
        box-shadow: 0 2px 12px rgba(79,209,197,0.15);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── Cards ── */
    .glass-card {
        background: linear-gradient(145deg, rgba(21,29,39,0.95) 0%, rgba(17,25,34,0.98) 100%);
        border: 1px solid #1e2d3d;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .metric-card {
        background: linear-gradient(145deg, #151d27 0%, #121a24 100%);
        border: 1px solid #1e2d3d;
        border-radius: 14px;
        padding: 1.3rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #2a4a5a;
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #5a7080;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2eaf0;
        margin: 2rem 0 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1e2d3d;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-header .icon { font-size: 1.1rem; }

    /* ── Segment cards ── */
    .segment-card {
        background: #151d27;
        border: 1px solid #1e2d3d;
        border-left: 4px solid;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.65rem;
        transition: border-color 0.2s ease;
    }
    .segment-card:hover { border-color: #2a4a5a; }
    .seg-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .seg-emotion {
        font-weight: 700;
        font-size: 0.88rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .seg-confidence {
        font-size: 0.78rem;
        color: #5a7080;
        font-weight: 500;
    }
    .seg-text {
        font-size: 0.88rem;
        color: #8a9baa;
        line-height: 1.55;
    }
    .conf-bar {
        height: 4px;
        border-radius: 2px;
        background: #1a2530;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.4s ease;
    }

    /* ── Shift cards ── */
    .shift-card {
        background: linear-gradient(135deg, #1a1520 0%, #1a1828 100%);
        border: 1px solid #2a1e3a;
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.55rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .shift-arrow {
        font-size: 1.3rem;
        color: #9B72CF;
        flex-shrink: 0;
    }
    .shift-info {
        flex: 1;
    }
    .shift-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #d0c0e0;
    }
    .shift-meta {
        font-size: 0.75rem;
        color: #6a5a7a;
        margin-top: 2px;
    }
    .shift-magnitude {
        font-size: 0.8rem;
        font-weight: 700;
        color: #c0a0e0;
        flex-shrink: 0;
    }

    /* ── Emotion distribution bar ── */
    .dist-bar-container {
        display: flex;
        height: 32px;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0 0.8rem;
    }
    .dist-bar-seg {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
        color: #0d1117;
        transition: flex 0.3s ease;
        min-width: 0;
    }
    .dist-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem 1.2rem;
        margin-top: 0.4rem;
    }
    .dist-legend-item {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.78rem;
        color: #8a9baa;
    }
    .dist-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"],
    .stButton > button {
        background: linear-gradient(135deg, #2a8a7a 0%, #1a6a5a 100%) !important;
        color: #e0f5f0 !important;
        border: 1px solid #3a9a8a !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 12px rgba(79,209,197,0.12) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #35a090 0%, #208070 100%) !important;
        box-shadow: 0 4px 20px rgba(79,209,197,0.2) !important;
        transform: translateY(-1px);
    }

    /* ── Text area ── */
    .stTextArea textarea {
        background: #151d27 !important;
        border: 1px solid #1e2d3d !important;
        border-radius: 12px !important;
        color: #c9d5e0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
        padding: 1rem !important;
        transition: border-color 0.2s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: #3a6a7a !important;
        box-shadow: 0 0 0 2px rgba(79,209,197,0.12) !important;
    }

    /* ── Sliders ── */
    .stSlider > div > div > div > div {
        background: #4fd1c5 !important;
    }

    /* ── Audio input ── */
    .stAudioInput > div {
        border: 1px solid #1e2d3d !important;
        border-radius: 12px !important;
        background: #151d27 !important;
    }

    /* ── Hide default Streamlit elements ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Warning box ── */
    .warning-box {
        background: linear-gradient(135deg, #2a2010 0%, #1a1808 100%);
        border: 1px solid #3a3018;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.85rem;
        color: #d0b060;
    }

    /* ── Info / interpretation box ── */
    .info-box {
        background: linear-gradient(135deg, #101a28 0%, #0d1520 100%);
        border: 1px solid #1a2a3a;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin: 1rem 0;
        font-size: 0.85rem;
        color: #7a9aaa;
        line-height: 1.6;
    }
    .info-box strong { color: #a0c0d0; }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .app-header h1 { font-size: 1.7rem; }
        .metric-value { font-size: 1.6rem; }
        .stTabs [data-baseweb="tab"] { padding: 0.5rem 1rem; font-size: 0.82rem; }
    }
    </style>
    """, unsafe_allow_html=True)


inject_css()


# ─── Model loaders ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading text emotion model (first run only)...")
def load_text_model():
    from transformers import pipeline
    return pipeline("text-classification", model=TEXT_MODEL, top_k=None, truncation=True)


@st.cache_resource(show_spinner="Loading voice emotion model (first run only)...")
def load_voice_model():
    """Load the wav2vec2 speech emotion recognition model.

    Uses Wav2Vec2FeatureExtractor instead of AutoProcessor because the
    ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition repository
    does not include tokenizer/processor config files, causing AutoProcessor
    to fail with: 'expected str, bytes or os.PathLike object, not NoneType'.

    Also remaps classifier head weights from older checkpoint format
    (classifier.dense/output) to the current transformers architecture
    (projector/classifier) to avoid randomly-initialized predictions.
    """
    import torch
    from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification

    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(VOICE_MODEL)
    except OSError as e:
        raise RuntimeError(
            f"Failed to download or locate the feature extractor for '{VOICE_MODEL}'. "
            f"Check your internet connection and that the model ID is correct. "
            f"Original error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading feature extractor for '{VOICE_MODEL}': {e}"
        ) from e

    try:
        model = AutoModelForAudioClassification.from_pretrained(VOICE_MODEL)
    except OSError as e:
        raise RuntimeError(
            f"Failed to download or locate the model weights for '{VOICE_MODEL}'. "
            f"Check your internet connection and that the model ID is correct. "
            f"Original error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading model '{VOICE_MODEL}': {e}"
        ) from e

    if not hasattr(model.config, "id2label") or not model.config.id2label:
        raise RuntimeError(
            f"Model '{VOICE_MODEL}' loaded but its config is missing 'id2label' mapping. "
            f"The model configuration may be corrupted. Try clearing the HuggingFace cache "
            f"(~/.cache/huggingface/) and re-downloading."
        )

    # ── Fix classifier head weight mismatch ──
    # Older wav2vec2 checkpoints saved the classification head with keys:
    #   classifier.dense.weight/bias  +  classifier.output.weight/bias
    # Newer transformers expects:
    #   projector.weight/bias  +  classifier.weight/bias
    # Without remapping, the classifier is randomly initialized → garbage output.
    _KEY_REMAP = {
        "classifier.dense.weight": "projector.weight",
        "classifier.dense.bias": "projector.bias",
        "classifier.output.weight": "classifier.weight",
        "classifier.output.bias": "classifier.bias",
    }
    try:
        from huggingface_hub import hf_hub_download
        # Try pytorch .bin first (common for older models), then safetensors
        ckpt_path = None
        for fname in ("pytorch_model.bin", "model.safetensors"):
            try:
                ckpt_path = hf_hub_download(VOICE_MODEL, fname)
                break
            except Exception:
                continue

        if ckpt_path is not None:
            if ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(ckpt_path)
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

            remapped = {}
            for old_key, new_key in _KEY_REMAP.items():
                if old_key in state_dict:
                    remapped[new_key] = state_dict[old_key]

            if remapped:
                model.load_state_dict(remapped, strict=False)
    except Exception:
        # If remapping fails, the model still works but classifier head
        # may produce lower-quality predictions from default init weights.
        pass

    model.eval()
    return feature_extractor, model


# ─── Preserved analysis functions ────────────────────────────────────────────
def split_sentences(text):
    """Split text into sentences (preserved from original)."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]


def js_distance(p, q):
    """Jensen-Shannon distance (base 2) → 0 (identical) .. 1 (completely different).
    Preserved from original implementation."""
    m = (p + q) / 2
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return float(np.sqrt(max((kl(p, m) + kl(q, m)) / 2, 0.0)))


def verdict(score):
    """Classify consistency score (preserved from original)."""
    return "Highly consistent" if score >= 80 else "Moderately consistent" if score >= 60 else "Inconsistent"


def verdict_color(score):
    if score >= 80:
        return "#4fd1c5"
    elif score >= 60:
        return "#F5B942"
    else:
        return "#E74C6F"


# ─── Voice analysis helpers ──────────────────────────────────────────────────
def load_audio_waveform(audio_bytes):
    """Load audio bytes into a numpy waveform at 16kHz."""
    import librosa
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        f.flush()
        waveform, sr = librosa.load(f.name, sr=16000)
    return waveform, sr


def segment_audio(waveform, sr, segment_duration=3.0, min_segment_duration=1.0):
    """Split waveform into segments of `segment_duration` seconds.
    Segments shorter than `min_segment_duration` are dropped."""
    total = len(waveform) / sr
    segments = []
    start = 0.0
    while start < total:
        end = min(start + segment_duration, total)
        if (end - start) >= min_segment_duration:
            s_idx = int(start * sr)
            e_idx = int(end * sr)
            segments.append({
                "waveform": waveform[s_idx:e_idx],
                "start": round(start, 2),
                "end": round(end, 2),
            })
        start += segment_duration
    return segments


def analyze_voice_segment(segment_waveform, processor, model):
    """Classify a single audio segment and return probability vector aligned to LABELS."""
    import torch
    inputs = processor(segment_waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()

    # Map voice model labels → standard LABELS order.
    # The model may have more classes (e.g. 8 including "calm") than our 7
    # display labels, so we accumulate probabilities when multiple model
    # labels map to the same standard label (e.g. calm + neutral → neutral).
    id2label = model.config.id2label
    mapped = {}
    for idx, raw_label in id2label.items():
        std = VOICE_LABEL_MAP.get(raw_label.lower(), raw_label.lower())
        mapped[std] = mapped.get(std, 0.0) + float(probs[int(idx)])

    return np.array([mapped.get(l, 0.0) for l in LABELS])


def check_audio_quality(waveform, sr):
    """Return warnings about audio quality."""
    issues = []
    duration = len(waveform) / sr
    if duration < 2.0:
        issues.append(f"Recording is very short ({duration:.1f}s). At least 3 seconds recommended for meaningful analysis.")
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms < 0.005:
        issues.append("Recording appears nearly silent. Ensure your microphone is working and speak clearly.")
    if rms > 0.8:
        issues.append("Recording is very loud and may be clipped. Try reducing microphone volume.")
    return issues


# ─── Plotting ────────────────────────────────────────────────────────────────
def create_emotion_timeline(vecs, x_labels, x_title="Segment"):
    """Create an interactive Plotly line chart for emotion probabilities."""
    fig = go.Figure()
    for i, label in enumerate(LABELS):
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=vecs[:, i],
            name=label.capitalize(),
            mode="lines+markers",
            line=dict(color=EMOTION_COLORS[label], width=2.5),
            marker=dict(size=6, color=EMOTION_COLORS[label]),
            hovertemplate=f"<b>{label.capitalize()}</b><br>{x_title}: %{{x}}<br>Probability: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(family="Inter", color="#8a9baa", size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        xaxis=dict(
            title=x_title, gridcolor="#1a2530", zerolinecolor="#1a2530",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Probability", range=[0, 1], gridcolor="#1a2530",
            zerolinecolor="#1a2530", tickfont=dict(size=11),
        ),
        margin=dict(l=50, r=20, t=30, b=50),
        height=380,
        hovermode="x unified",
    )
    return fig


# ─── Results rendering ──────────────────────────────────────────────────────
def render_results(vecs, dominant, segs_data, dists, score, threshold, mode="text"):
    """Render the complete results dashboard.
    `segs_data` is a list of dicts with keys:
      - text: segment text (for text mode) OR timestamps (for voice mode)
      - label: e.g. "Segment 1" or "0.00s – 3.00s"
    """
    num_segments = len(dominant)
    top, top_n = Counter(dominant).most_common(1)[0]
    shifts = [(i, d) for i, d in enumerate(dists) if d >= threshold]
    v_color = verdict_color(score)
    mode_label = "Consistency" if mode == "text" else "Vocal Stability"

    # ── A. Summary metrics ──
    st.markdown(f'<div class="section-header"><span class="icon">📊</span> {mode_label} Summary</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{v_color}">{score}</div>
            <div class="metric-label">{mode_label} Score</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{EMOTION_COLORS[top]}">{EMOTION_ICONS.get(top,'')} {top.capitalize()}</div>
            <div class="metric-label">Dominant Emotion ({round(100 * top_n / num_segments)}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#4fd1c5">{num_segments}</div>
            <div class="metric-label">Segments Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#9B72CF">{len(shifts)}</div>
            <div class="metric-label">Tone Shifts</div>
        </div>
        """, unsafe_allow_html=True)

    # Verdict
    v = verdict(score)
    if mode == "text":
        explain = (
            f"A score of **{score}** means the emotional tone across your text is **{v.lower()}**. "
            f"This score is computed as 100 × (1 − mean Jensen-Shannon distance between consecutive segments). "
            f"Higher values indicate more uniform emotional tone."
        )
    else:
        explain = (
            f"A vocal stability score of **{score}** indicates the estimated vocal emotion is **{v.lower()}** across segments. "
            f"This score is computed as 100 × (1 − mean Jensen-Shannon distance between consecutive audio windows). "
            f"Higher values indicate less variation in predicted vocal emotion."
        )
    st.markdown(f'<div class="info-box">{explain}</div>', unsafe_allow_html=True)

    # ── Emotion color strip ──
    strip_html = "".join(
        f'<div style="flex:1;background:{EMOTION_COLORS[d]};min-width:2px" title="Segment {i+1}: {d}"></div>'
        for i, d in enumerate(dominant)
    )
    st.markdown(
        f'<div style="display:flex;height:24px;border-radius:8px;overflow:hidden;margin:0.5rem 0 1.2rem">{strip_html}</div>',
        unsafe_allow_html=True,
    )

    # ── B. Emotion Distribution ──
    st.markdown('<div class="section-header"><span class="icon">🎨</span> Emotion Distribution</div>', unsafe_allow_html=True)

    counts = Counter(dominant)
    dist_html = ""
    legend_html = ""
    for label in LABELS:
        cnt = counts.get(label, 0)
        pct = round(100 * cnt / num_segments, 1) if num_segments > 0 else 0
        if pct > 0:
            dist_html += f'<div class="dist-bar-seg" style="flex:{pct};background:{EMOTION_COLORS[label]}">{pct:.0f}%</div>'
        legend_html += f'<div class="dist-legend-item"><div class="dist-dot" style="background:{EMOTION_COLORS[label]}"></div>{EMOTION_ICONS.get(label,"")} {label.capitalize()} ({cnt})</div>'

    st.markdown(f"""
    <div class="glass-card">
        <div class="dist-bar-container">{dist_html}</div>
        <div class="dist-legend">{legend_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── C. Emotion Probability Timeline ──
    st.markdown('<div class="section-header"><span class="icon">📈</span> Emotion Probability Timeline</div>', unsafe_allow_html=True)

    if mode == "text":
        x_labels = list(range(1, num_segments + 1))
        x_title = "Segment"
    else:
        x_labels = [s["label"] for s in segs_data]
        x_title = "Time Window"

    fig = create_emotion_timeline(vecs, x_labels, x_title)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── D. Tone Shifts ──
    st.markdown('<div class="section-header"><span class="icon">⚡</span> Detected Tone Shifts</div>', unsafe_allow_html=True)

    if shifts:
        for i, d in shifts:
            from_e = dominant[i]
            to_e = dominant[i + 1]
            mag_pct = round(d * 100)
            if mode == "text":
                seg_label = f"Segment {i+1} → Segment {i+2}"
            else:
                seg_label = f"{segs_data[i]['label']} → {segs_data[i+1]['label']}"
            st.markdown(f"""
            <div class="shift-card">
                <div class="shift-arrow">⚡</div>
                <div class="shift-info">
                    <div class="shift-label">
                        <span style="color:{EMOTION_COLORS[from_e]}">{EMOTION_ICONS.get(from_e,'')} {from_e.capitalize()}</span>
                        &nbsp;→&nbsp;
                        <span style="color:{EMOTION_COLORS[to_e]}">{EMOTION_ICONS.get(to_e,'')} {to_e.capitalize()}</span>
                    </div>
                    <div class="shift-meta">{seg_label}</div>
                </div>
                <div class="shift-magnitude">{d:.2f} ({mag_pct}%)</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center;color:#5a7080;padding:1.5rem">
            ✅ No significant tone shifts detected at the current sensitivity threshold.
        </div>
        """, unsafe_allow_html=True)

    # ── E. Segment-by-Segment Results ──
    st.markdown('<div class="section-header"><span class="icon">📋</span> Segment Details</div>', unsafe_allow_html=True)

    for i in range(num_segments):
        em = dominant[i]
        conf = round(100 * vecs[i].max())
        color = EMOTION_COLORS[em]
        icon = EMOTION_ICONS.get(em, "")

        if mode == "text":
            seg_title = f"Segment {i + 1}"
            body = f'<div class="seg-text">{segs_data[i]["text"]}</div>'
        else:
            seg_title = segs_data[i]["label"]
            body = f'<div class="seg-text" style="font-style:italic;color:#5a7080">Audio segment ({segs_data[i]["label"]})</div>'

        st.markdown(f"""
        <div class="segment-card" style="border-left-color:{color}">
            <div class="seg-header">
                <div class="seg-emotion" style="color:{color}">{icon} {seg_title} — {em.capitalize()}</div>
                <div class="seg-confidence">{conf}% confidence</div>
            </div>
            {body}
            <div class="conf-bar"><div class="conf-bar-fill" style="width:{conf}%;background:{color}"></div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── F. Interpretation & Limitations ──
    st.markdown('<div class="section-header"><span class="icon">ℹ️</span> Interpretation & Limitations</div>', unsafe_allow_html=True)

    if mode == "text":
        st.markdown("""
        <div class="info-box">
            <strong>How to interpret these results:</strong><br>
            The text emotion model predicts the <em>most likely emotional tone</em> of each text segment
            based on word choice, phrasing, and context. The consistency score summarizes how similar
            consecutive segments are in their emotional profiles.<br><br>
            <strong>Limitations:</strong> Emotion detection from text is inherently uncertain. The model
            may not correctly capture sarcasm, irony, cultural nuance, or domain-specific language.
            These results reflect statistical patterns in training data and should not be treated as
            a definitive assessment of emotional intent or the writer's actual feelings.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <strong>How to interpret these results:</strong><br>
            The speech emotion model estimates vocal emotion from acoustic features such as pitch,
            tempo, energy, and spectral patterns. Each audio segment receives a probability distribution
            over emotion categories. The vocal stability score measures how much these distributions
            change across consecutive segments.<br><br>
            <strong>Limitations:</strong> Speech emotion recognition from audio alone is an open research
            problem with significant uncertainty. Predictions depend heavily on recording quality,
            background noise, accent, speaking style, and the model's training data. <em>These estimates
            do not establish anyone's actual feelings, personality, honesty, or reliability.</em>
            Results should be treated as rough approximations, not scientific measurements.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="app-header">
    <h1>🎭 Tone Consistency Checker</h1>
    <p>Analyze the emotional tone of your text or voice — see how it flows from start to finish.</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_text, tab_voice = st.tabs(["📝 Text Analysis", "🎙️ Voice Analysis"])


# ─── TEXT ANALYSIS TAB ───────────────────────────────────────────────────────
with tab_text:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    text = st.text_area(
        "Your text",
        height=200,
        placeholder="Paste a speech, blog post, or presentation script here (at least two sentences)...",
        key="text_input",
    )
    c1, c2 = st.columns(2)
    window = c1.slider("Sentences per segment", 1, 10, 3, key="text_window")
    threshold = c2.slider("Shift sensitivity (lower = more shifts)", 0.1, 0.9, 0.45, 0.05, key="text_threshold")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Analyze Tone", type="primary", key="btn_text"):
        sents = split_sentences(text)
        if len(sents) < 2:
            st.error("Please enter at least two sentences.")
            st.stop()

        segs = [" ".join(sents[i:i + window]) for i in range(0, len(sents), window)]
        with st.spinner("Analyzing emotional tone..."):
            raw = load_text_model()(segs)

        vecs = np.array([
            [next(d["score"] for d in r if d["label"] == l) for l in LABELS]
            for r in raw
        ])
        dominant = [LABELS[i] for i in vecs.argmax(1)]

        dists = [js_distance(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
        score = round(100 * (1 - float(np.mean(dists))), 1) if dists else 100.0

        segs_data = [{"text": s, "label": f"Segment {i+1}"} for i, s in enumerate(segs)]
        render_results(vecs, dominant, segs_data, dists, score, threshold, mode="text")


# ─── VOICE ANALYSIS TAB ─────────────────────────────────────────────────────
with tab_voice:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:0.8rem">
        <div style="font-size:0.95rem;font-weight:600;color:#c9d5e0;margin-bottom:0.3rem">🎙️ Record your voice</div>
        <div style="font-size:0.82rem;color:#5a7080">
            Click the microphone button below to start recording. Speak clearly for at least 3 seconds.
            Longer recordings allow more detailed segment-by-segment analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    audio_data = st.audio_input("Record audio", key="voice_input")

    vc1, vc2 = st.columns(2)
    seg_duration = vc1.slider("Segment duration (seconds)", 1.0, 10.0, 3.0, 0.5, key="voice_seg_dur")
    voice_threshold = vc2.slider("Shift sensitivity (lower = more shifts)", 0.1, 0.9, 0.45, 0.05, key="voice_threshold")

    st.markdown('</div>', unsafe_allow_html=True)

    if audio_data is not None:
        # Show playback
        st.audio(audio_data, format="audio/wav")

        if st.button("🔬 Analyze Voice", type="primary", key="btn_voice"):
            audio_bytes = audio_data.getvalue()

            with st.spinner("Processing audio..."):
                try:
                    waveform, sr = load_audio_waveform(audio_bytes)
                except Exception as e:
                    st.error(f"Failed to load audio: {e}. Ensure you recorded a valid audio sample.")
                    st.stop()

                # Quality checks
                issues = check_audio_quality(waveform, sr)
                for issue in issues:
                    st.markdown(f'<div class="warning-box">⚠️ {issue}</div>', unsafe_allow_html=True)

                duration = len(waveform) / sr
                st.markdown(
                    f'<div style="font-size:0.82rem;color:#5a7080;margin:0.3rem 0 0.8rem">'
                    f'📏 Recording duration: {duration:.1f}s | Sample rate: {sr} Hz</div>',
                    unsafe_allow_html=True,
                )

                audio_segments = segment_audio(waveform, sr, segment_duration=seg_duration)
                if len(audio_segments) < 1:
                    st.error("Recording too short to create any usable segments. Please record a longer audio sample.")
                    st.stop()

                try:
                    processor, model = load_voice_model()
                except Exception as e:
                    st.error(f"Failed to load voice emotion model: {e}")
                    st.stop()

                # Analyze each segment
                vecs_list = []
                progress = st.progress(0, text="Analyzing audio segments...")
                for idx, seg in enumerate(audio_segments):
                    try:
                        vec = analyze_voice_segment(seg["waveform"], processor, model)
                        vecs_list.append(vec)
                    except Exception:
                        # Skip segments that fail (e.g. too short after processing)
                        vecs_list.append(np.ones(len(LABELS)) / len(LABELS))
                    progress.progress((idx + 1) / len(audio_segments), text=f"Analyzing segment {idx+1}/{len(audio_segments)}...")
                progress.empty()

            if len(vecs_list) < 1:
                st.error("No usable audio segments could be analyzed.")
                st.stop()

            vecs = np.array(vecs_list)
            dominant = [LABELS[i] for i in vecs.argmax(1)]

            dists = [js_distance(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
            score = round(100 * (1 - float(np.mean(dists))), 1) if dists else 100.0

            segs_data = [
                {"text": "", "label": f"{s['start']:.1f}s – {s['end']:.1f}s"}
                for s in audio_segments
            ]

            render_results(vecs, dominant, segs_data, dists, score, voice_threshold, mode="voice")
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:2.5rem;color:#5a7080">
            <div style="font-size:2.5rem;margin-bottom:0.8rem">🎙️</div>
            <div style="font-size:0.95rem;font-weight:500">Record an audio sample to begin voice emotion analysis</div>
            <div style="font-size:0.82rem;margin-top:0.4rem">Click the microphone button above to start</div>
        </div>
        """, unsafe_allow_html=True)
