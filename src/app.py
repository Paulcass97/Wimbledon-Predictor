import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Wimbledon Predictor",
    page_icon="🎾",
    layout="wide"
)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'wimbledon_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

bundle = load_model()
model  = bundle['model']
scaler = bundle['scaler']
feature_cols = bundle['features']

# ── Header ───────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#006400;'>
        🎾 Wimbledon Match Predictor
    </h1>
    <p style='text-align:center; color:gray;'>
        Built on grass court ATP data 2000–2025 · Random Forest · AUC 0.71
    </p>
    <hr>
""", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🟢 Player 1")
    p1_rank         = st.number_input("ATP Rank", min_value=1, max_value=500, value=1, key='p1_rank')
    p1_elo          = st.number_input("Grass Elo", min_value=1000, max_value=2500, value=1800, key='p1_elo')
    p1_grass_win_pct = st.slider("Grass Win %", 0.0, 1.0, 0.65, 0.01, key='p1_gwp')
    p1_form5        = st.slider("Last 5 Grass Form", 0.0, 1.0, 0.6, 0.1, key='p1_f5')
    p1_form10       = st.slider("Last 10 Overall Form", 0.0, 1.0, 0.6, 0.1, key='p1_f10')

with col3:
    st.markdown("### 🔴 Player 2")
    p2_rank         = st.number_input("ATP Rank", min_value=1, max_value=500, value=5, key='p2_rank')
    p2_elo          = st.number_input("Grass Elo", min_value=1000, max_value=2500, value=1700, key='p2_elo')
    p2_grass_win_pct = st.slider("Grass Win %", 0.0, 1.0, 0.55, 0.01, key='p2_gwp')
    p2_form5        = st.slider("Last 5 Grass Form", 0.0, 1.0, 0.4, 0.1, key='p2_f5')
    p2_form10       = st.slider("Last 10 Overall Form", 0.0, 1.0, 0.5, 0.1, key='p2_f10')

with col2:
    st.markdown("### ⚙️ Match Context")
    round_choice = st.selectbox("Round", 
        ['R128','R64','R32','R16','QF','SF','Final'])
    rest_diff    = st.slider("Player 1 rest advantage (days)", -10, 10, 0)
    is_wimbledon = st.toggle("Wimbledon match?", value=True)

    round_map = {'R128':1,'R64':2,'R32':3,'R16':4,'QF':5,'SF':6,'Final':7}
    month_val  = 6  # Wimbledon is always June/July
    
    features = pd.DataFrame([[
        p2_rank - p1_rank,
        p1_elo  - p2_elo,
        p1_grass_win_pct - p2_grass_win_pct,
        p1_form5  - p2_form5,
        p1_form10 - p2_form10,
        rest_diff,
        np.sin(2 * np.pi * month_val / 12),
        np.cos(2 * np.pi * month_val / 12),
        int(is_wimbledon),
        round_map[round_choice]
    ]], columns=feature_cols)

    if scaler:
        features_input = scaler.transform(features)
    else:
        features_input = features.values

    prob_p1 = model.predict_proba(features_input)[0][1]
    prob_p2 = 1 - prob_p1

    st.markdown("---")
    st.markdown("### 📊 Win Probability")

    # Progress bars
    st.markdown(f"**🟢 Player 1: {prob_p1*100:.1f}%**")
    st.progress(prob_p1)
    st.markdown(f"**🔴 Player 2: {prob_p2*100:.1f}%**")
    st.progress(prob_p2)

    st.markdown("---")
    if prob_p1 >= 0.6:
        st.success(f"🟢 Player 1 is the clear favourite")
    elif prob_p2 >= 0.6:
        st.error(f"🔴 Player 2 is the clear favourite")
    else:
        st.warning(f"⚖️ This is a closely contested match")

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray; font-size:12px;'>Built by Paul Cassady · MSc Data Science · Nottingham Trent University · 2025</p>", 
            unsafe_allow_html=True)