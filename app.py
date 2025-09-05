# streamlit_app.py — simple, backend-agnostic UI
import os, time
import streamlit as st
from translator import load_translator  # uses backend.json inside best_model/

st.set_page_config(page_title="CN→EN Translator", page_icon="🌐", layout="centered")
st.title("CN → EN Translator")

RUN_DIR = os.environ.get("RUN_DIR", "run")  # e.g., runs/jen_marian
try:
    lr = load_translator(RUN_DIR, device="cpu")  # CPU by default; change to None to auto-pick GPU/CPU
    st.caption(f"Loaded **[{lr.backend}]** from `{RUN_DIR}/best_model`")
except Exception as e:
    st.error(f"Failed to load model from `{RUN_DIR}/best_model`.\n"
             f"Set RUN_DIR env var or fix the folder. Details:\n\n{e}")
    st.stop()

# --- Model type selection ---
model_type = st.selectbox(
    "Choose NLP Model Type",
    ("SMT (IBM1)", "NMT (Transformer)", "Hybrid SMT-NMT Model"),
    index=1  # Default to NMT
)

# Map model type to folder name
model_folder_map = {
    "SMT (IBM1)": "best_model_smt",
    "NMT (Transformer)": "best_model_nmt",
    "Hybrid SMT-NMT Model": "best_model_hybrid"
}
model_folder = model_folder_map[model_type]
model_path = os.path.join(RUN_DIR, model_folder)

if not os.path.isdir(model_path):
    st.error(f"Model folder `{model_path}` not found. Please train or place the model first.")
    st.stop()

try:
    lr = load_translator(model_path, device="cpu")
    st.caption(f"Loaded **[{lr.backend}]** from `{model_path}`")
except Exception as e:
    st.error(f"Failed to load model from `{model_path}`.\nDetails:\n\n{e}")
    st.stop()

cn = st.text_area("Chinese", height=160, placeholder="输入中文句子…")
if st.button("Translate 🚀", type="primary"):
    if not cn.strip():
        st.warning("Please enter a sentence.")
    else:
        t0 = time.time()
        out, info = lr.translator.translate(
            cn,
            num_beams=5,
            max_new_tokens=128,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            return_info=True,
            model_type=model_type
        )
        ms = (time.time() - t0) * 1000
        st.markdown("**English**")
        st.text_area("Output", out, height=160, key="output")
        st.caption(f"{info} • ⏱️ {ms:.0f} ms")
