"""Emotional Tone Consistency Checker - Streamlit version.
Run locally:  streamlit run streamlit_app.py
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from transformers import pipeline

MODEL = "j-hartmann/emotion-english-distilroberta-base"
LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
COLORS = ["#c0392b", "#7d8f2a", "#6c4a9e", "#e0a100", "#8a949a", "#2e6fb5", "#d9682b"]
COLOR_OF = dict(zip(LABELS, COLORS))


@st.cache_resource(show_spinner="Loading emotion model (first run only)...")
def load_model():
    return pipeline("text-classification", model=MODEL, top_k=None, truncation=True)


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]


def js_distance(p, q):
    m = (p + q) / 2
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return float(np.sqrt(max((kl(p, m) + kl(q, m)) / 2, 0.0)))


def verdict(score):
    return "Highly consistent" if score >= 80 else "Moderately consistent" if score >= 60 else "Inconsistent"


st.set_page_config(page_title="Tone Consistency Checker", page_icon="🎭")
st.title("Tone consistency checker")
st.caption("Paste a speech, blog post or presentation script to see how its emotional tone moves from start to finish.")

text = st.text_area("Your text", height=220, placeholder="Paste at least two sentences...")
c1, c2 = st.columns(2)
window = c1.slider("Sentences per segment", 1, 10, 3)
threshold = c2.slider("Shift sensitivity (lower = more shifts)", 0.1, 0.9, 0.45, 0.05)

if st.button("Analyze tone", type="primary"):
    sents = split_sentences(text)
    if len(sents) < 2:
        st.error("Please enter at least two sentences.")
        st.stop()

    segs = [" ".join(sents[i:i + window]) for i in range(0, len(sents), window)]
    with st.spinner("Analyzing..."):
        raw = load_model()(segs)
    vecs = np.array([[next(d["score"] for d in r if d["label"] == l) for l in LABELS] for r in raw])
    dominant = [LABELS[i] for i in vecs.argmax(1)]

    dists = [js_distance(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
    score = round(100 * (1 - float(np.mean(dists))), 1) if dists else 100.0
    top, top_n = Counter(dominant).most_common(1)[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Consistency score", score)
    m2.metric("Verdict", verdict(score))
    m3.metric("Main emotion", f"{top} ({round(100 * top_n / len(dominant))}%)")

    strip = "".join(f'<div style="flex:1;background:{COLOR_OF[d]}" title="Segment {i + 1}: {d}"></div>'
                    for i, d in enumerate(dominant))
    st.markdown(f'<div style="display:flex;height:22px;border-radius:4px;overflow:hidden;margin:12px 0">{strip}</div>',
                unsafe_allow_html=True)

    df = pd.DataFrame(vecs, columns=LABELS, index=range(1, len(segs) + 1))
    df.index.name = "Segment"
    st.line_chart(df, color=COLORS)

    st.subheader("Tone shifts")
    shifts = [(i, d) for i, d in enumerate(dists) if d >= threshold]
    if shifts:
        for i, d in shifts:
            st.warning(f"Segment {i + 1} to {i + 2}: {dominant[i]} to {dominant[i + 1]} (magnitude {d:.2f})")
    else:
        st.info("No sharp shifts found.")

    st.subheader("Segments")
    for i, s in enumerate(segs):
        conf = round(100 * vecs[i].max())
        st.markdown(f"**Segment {i + 1}: :{'gray'}[{dominant[i]} ({conf}%)]**  \n{s}")
