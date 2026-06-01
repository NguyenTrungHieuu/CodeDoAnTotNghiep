# Streamlit demo — shares model + data helpers with flask_msn_site (co_naml_infer.py).

import warnings
from pathlib import Path

import streamlit as st
import torch

from co_naml_infer import SimpleCoNAML, _load_news_table, encode_news_batch

warnings.filterwarnings("ignore")
st.set_page_config(page_title="News Recommendation", page_icon="📰", layout="wide")


@st.cache_resource
def load_data():
    base = Path(__file__).resolve().parent
    return _load_news_table(base)


# ============================================================
# MAIN APP
# ============================================================
st.title("📰 News Recommendation System")
st.markdown("### Co-NAML-LSTUR Demo | MIND-Small Dataset")
st.markdown("---")

# Initialize session state
if "history_ids" not in st.session_state:
    st.session_state.history_ids = []

# Load data
all_news, config = load_data()
st.success(f"✅ Loaded {len(all_news):,} news articles")

# Initialize model
device = torch.device("cpu")
model = SimpleCoNAML(config).to(device)
model.eval()
ckpt = Path(__file__).resolve().parent / "best_co_naml_lstur.pt"
if ckpt.is_file():
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    st.caption("Loaded weights from `best_co_naml_lstur.pt` (strict=False).")
# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Number of Recommendations", 3, 15, 5)
    
    st.markdown("---")
    st.header("📚 Select News (History)")
    
    sample = all_news.head(50)
    for _, row in sample.iterrows():
        nid = row['news_id']
        checked = nid in st.session_state.history_ids
        if st.checkbox(f"{row['title'][:70]}...", checked, key=f"cb_{nid}"):
            if nid not in st.session_state.history_ids:
                st.session_state.history_ids.append(nid)
        else:
            if nid in st.session_state.history_ids:
                st.session_state.history_ids.remove(nid)
    
    st.markdown("---")
    custom_ids = st.text_input("Add News IDs", placeholder="N12345, N67890")
    if st.button("➕ Add"):
        for nid in [x.strip() for x in custom_ids.split(",") if x.strip()]:
            if nid in all_news['news_id'].values and nid not in st.session_state.history_ids:
                st.session_state.history_ids.append(nid)
    
    if st.button("🗑️ Clear All"):
        st.session_state.history_ids = []
        st.rerun()
    
    st.info(f"Selected: {len(st.session_state.history_ids)} articles")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.header("🎯 Recommendations")
    
    if st.button("🔄 Generate", type="primary", use_container_width=True):
        if not st.session_state.history_ids:
            st.warning("Please select at least 1 article!")
        else:
            with st.spinner("Generating recommendations..."):
                with torch.no_grad():
                    # Encode history
                    history_rows = all_news[all_news['news_id'].isin(st.session_state.history_ids)]
                    if len(history_rows) == 0:
                        st.error("No valid history found")
                    else:
                        h_vecs = encode_news_batch(model, history_rows)  # [H, D]
                        
                        # Sample candidates
                        cands = all_news[~all_news['news_id'].isin(st.session_state.history_ids)]
                        if len(cands) > 100:
                            cands = cands.sample(100)
                        
                        c_vecs = encode_news_batch(model, cands)  # [C, D]
                        
                        # Get scores
                        scores = model(h_vecs, c_vecs)  # [1, C]
                        probs = torch.sigmoid(scores).squeeze(0).numpy()
                        
                        # Top K
                        k_actual = min(k, len(probs))
                        top_idx = probs.argsort()[-k_actual:][::-1]
                        
                        st.success(f"Top {k_actual} recommendations:")
                        
                        for rank, idx in enumerate(top_idx):
                            row = cands.iloc[idx]
                            score = float(probs[idx])
                            score_color = "#28a745" if score > 0.5 else "#ffc107"
                            
                            st.markdown(f"""
                            <div style="padding:15px; border:1px solid #e0e0e0; border-radius:10px; margin:10px 0; 
                                        background: linear-gradient(135deg, #f8f9fa, #e9ecef);">
                                <div style="display:flex; justify-content:space-between;">
                                    <h4 style="color:#1a73e8;">#{rank+1} {row['title']}</h4>
                                    <span style="background:{score_color}; color:white; padding:4px 12px; 
                                                 border-radius:20px; font-weight:bold;">
                                        {score:.3f}
                                    </span>
                                </div>
                                <p><b>{row['category']}</b> | {row['subcategory']}</p>
                                <p style="color:#555;">{str(row['abstract'])[:200]}...</p>
                            </div>
                            """, unsafe_allow_html=True)

with col2:
    st.header("📖 History")
    for nid in st.session_state.history_ids:
        r = all_news[all_news['news_id'] == nid]
        if len(r) > 0:
            r = r.iloc[0]
            st.markdown(f"""
            <div style="padding:8px; border-left:3px solid #1a73e8; margin:4px 0; background:#f0f7ff; border-radius:0 8px 8px 0;">
                <b>{r['title'][:80]}</b><br>
                <small>{r['category']}</small>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#888;">Co-NAML-LSTUR Demo | Streamlit</div>', unsafe_allow_html=True)