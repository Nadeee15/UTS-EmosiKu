import streamlit as st
import torch
import re
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import time

st.set_page_config(page_title="EmosiKu — Sistem Deteksi Psikologis", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Dekorasi daun latar belakang
st.markdown('''
<div class="leaf-deco leaf-1"></div>
<div class="leaf-deco leaf-2"></div>
<div class="leaf-deco leaf-3"></div>
''', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = "indobenchmark/indobert-base-p1"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_stopword():
    return StopWordRemoverFactory().create_stop_word_remover()

tokenizer, model = load_model()
stopword = load_stopword()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text, flags=re.MULTILINE).lower()
    return re.sub(r'\s+', ' ', stopword.remove(text)).strip()

def predict(text):
    inputs = tokenizer(clean_text(text), return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        out = model(**inputs)
    probs = torch.softmax(out.logits, dim=-1)[0]
    pred = torch.argmax(out.logits, dim=-1).item()
    return pred, probs[pred].item(), probs.numpy()

if 'history' not in st.session_state:
    st.session_state['history'] = []

# ── BRAND BAR ──────────────────────────────────────────────────────────
st.markdown('''
<div class="brand-row">
    <div class="brand-mark">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M10 3 C6 3 3 7 4 11 C5 15 10 18 10 18 C10 18 15 15 16 11 C17 7 14 3 10 3Z"
                  fill="white" opacity="0.9"/>
        </svg>
    </div>
    <div class="brand-name">EmosiKu<span>.</span></div>
    <div class="status-pill">
        <span class="status-dot"></span>IndoBERT Aktif
    </div>
</div>
''', unsafe_allow_html=True)

# ── TITLE ──────────────────────────────────────────────────────────────
st.markdown('''
<div class="title-container">
    <div class="title-eyebrow">sistem analisis klinis · kesehatan mental</div>
    <div class="title-main">Memahami <em>perasaan</em><br>melalui kata-kata.</div>
    <div class="title-sub">Deteksi Psikologis Berbasis Natural Language Processing — Bahasa Indonesia</div>
</div>
''', unsafe_allow_html=True)

# ── MAIN COLUMNS ───────────────────────────────────────────────────────
col1, col2 = st.columns([1.3, 1], gap="large")

with col1:
    st.markdown('''
    <div class="metric-card">
        <div class="card-label">Teks Evaluasi</div>
    ''', unsafe_allow_html=True)

    user_input = st.text_area(
        "",
        height=195,
        placeholder="Ceritakan apa yang dirasakan... sistem akan menganalisis pola bahasa secara otomatis.",
        label_visibility="collapsed"
    )

    if st.button("Analisis Teks"):
        if user_input:
            progress_bar = st.progress(0)
            status_text = st.empty()

            for msg, pct in [
                ("Membaca pola bahasa...", 25),
                ("Mengekstrak fitur leksikal...", 60),
                ("Menjalankan model NLP...", 90),
                ("Menyusun hasil analisis...", 100),
            ]:
                status_text.markdown(
                    f"<p style='font-family:DM Mono,monospace; color:#4a7c59; font-size:0.78rem; letter-spacing:1px; font-style:italic;'>● {msg}</p>",
                    unsafe_allow_html=True
                )
                progress_bar.progress(pct)
                time.sleep(0.45)

            status_text.empty()
            progress_bar.empty()

            pred, conf, probs = predict(user_input)
            st.session_state['history'].append({
                "Waktu": datetime.now().strftime("%H:%M"),
                "Cuplikan Teks": user_input[:48] + "...",
                "Status": "Terindikasi" if pred == 1 else "Normal",
                "Skor": f"{conf:.1%}"
            })

            with col2:
                st.markdown('<div class="metric-card"><div class="card-label">Hasil Diagnostik</div>', unsafe_allow_html=True)

                # Animasi nafas
                st.markdown('''
                <div class="breath-ring">
                    <div class="breath-inner"></div>
                </div>
                ''', unsafe_allow_html=True)

                if pred == 1:
                    st.markdown(f'''
                    <div class="alert-box">
                        <h2>Terindikasi</h2>
                        <p>Pola bahasa menunjukkan kemungkinan adanya gejala kecemasan atau depresi.</p>
                        <div class="conf-badge">{conf:.1%}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="safe-box">
                        <h2>Kondisi Normal</h2>
                        <p>Tidak ditemukan pola leksikal yang mengindikasikan gangguan psikologis signifikan.</p>
                        <div class="conf-badge">{conf:.1%}</div>
                    </div>
                    ''', unsafe_allow_html=True)

                st.markdown('<div class="dist-label">Distribusi Probabilitas</div>', unsafe_allow_html=True)
                st.progress(float(probs[0]), text=f"Kondisi Normal  {probs[0]:.1%}")
                st.progress(float(probs[1]), text=f"Indikasi Klinis  {probs[1]:.1%}")

                st.markdown('''
                <div class="insight-strip">
                    <span style="font-size:18px;">🌿</span>
                    <span><strong style="color:#4a7c59; font-weight:500;">Catatan penting:</strong>
                    Hasil ini bukan diagnosis resmi. Disarankan untuk berkonsultasi dengan tenaga psikologi atau psikiater profesional.</span>
                </div>
                ''', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.warning("Silakan masukkan teks terlebih dahulu untuk dianalisis.")

    st.markdown('</div>', unsafe_allow_html=True)

# ── LOG ─────────────────────────────────────────────────────────────────
if st.session_state['history']:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'''
    <div class="metric-card" style="padding:20px 28px 4px;">
        <div class="card-label">
            Log Sesi
            <span style="margin-left:auto; font-size:11px; color:#8a9e90;">{len(st.session_state["history"])} entri</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(st.session_state['history']),
        use_container_width=True,
        hide_index=True
    )