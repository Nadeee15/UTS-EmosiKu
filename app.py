import streamlit as st
import torch
import re
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import time

st.set_page_config(page_title="Sistem Deteksi Klinis", layout="wide")

# Membaca file CSS eksternal
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Elemen Bergerak (Floating Orbs Latar Belakang)
st.markdown('<div class="orb-1"></div><div class="orb-2"></div><div class="orb-3"></div>', unsafe_allow_html=True)

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
    with torch.no_grad(): out = model(**inputs)
    probs = torch.softmax(out.logits, dim=-1)[0]
    pred = torch.argmax(out.logits, dim=-1).item()
    return pred, probs[pred].item(), probs.numpy()

if 'history' not in st.session_state: st.session_state['history'] = []

st.markdown('<div class="title-container"><div class="title-main">SISTEM ANALISIS KLINIS</div><div style="color: #64748b; font-size: 1.2rem; letter-spacing: 1px; font-weight: 500;">Deteksi Kelainan Psikologis Berbasis Natural Language Processing</div></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown('<div class="metric-card"><h3 style="margin-bottom: 15px; color: #3b82f6;">Input Teks Evaluasi</h3>', unsafe_allow_html=True)
    user_input = st.text_area("", height=220, placeholder="Masukkan transkrip teks pasien atau data teks mentah untuk dievaluasi oleh sistem AI...", label_visibility="collapsed")
    
    if st.button("PROSES ANALISIS"):
        if user_input:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.markdown("<p style='color: #6366f1; font-style: italic; font-weight: 600;'>Mengkalibrasi model komputasi...</p>", unsafe_allow_html=True)
            time.sleep(0.5)
            progress_bar.progress(30)
            
            status_text.markdown("<p style='color: #6366f1; font-style: italic; font-weight: 600;'>Mengekstrak fitur leksikal psikologis...</p>", unsafe_allow_html=True)
            time.sleep(0.5)
            progress_bar.progress(70)
            
            status_text.markdown("<p style='color: #6366f1; font-style: italic; font-weight: 600;'>Menjalankan inferensi Deep Learning...</p>", unsafe_allow_html=True)
            time.sleep(0.6)
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            pred, conf, probs = predict(user_input)
            st.session_state['history'].append({"Waktu": datetime.now().strftime("%H:%M:%S"), "Input": user_input[:40]+"...", "Status": "Terindikasi" if pred==1 else "Normal", "Probabilitas": f"{conf:.2%}"})
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("<h3 style='color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; margin-bottom: 20px;'>Laporan Diagnostik Sistem</h3>", unsafe_allow_html=True)
                
                if pred == 1:
                    st.markdown('<div class="alert-box"><h2 style="color: #e11d48; margin:0;">TERINDIKASI</h2><p style="color:#be123c; margin-top:8px; font-size:1.1rem; font-weight: 500;">Sistem mendeteksi probabilitas tinggi adanya kelainan psikologis (Kecemasan/Depresi).</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-box"><h2 style="color: #059669; margin:0;">NORMAL</h2><p style="color:#047857; margin-top:8px; font-size:1.1rem; font-weight: 500;">Pola leksikal tidak menunjukkan indikasi kelainan psikologis yang signifikan.</p></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<p style='color: #64748b; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;'>Distribusi Probabilitas Kelas</p>", unsafe_allow_html=True)
                st.progress(float(probs[0]), text=f"Kondisi Normal ({probs[0]:.1%})")
                st.progress(float(probs[1]), text=f"Indikasi Klinis ({probs[1]:.1%})")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Input data tidak valid. Silakan masukkan teks.")
    st.markdown('</div>', unsafe_allow_html=True)

if len(st.session_state['history']) > 0:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><h3 style="color: #3b82f6; margin-bottom:15px;">Log Aktivitas Sistem Terpusat</h3></div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(st.session_state['history']), use_container_width=True)
