import json, pandas as pd, streamlit as st
from pathlib import Path
from PIL import Image
from model import MODELS, load, predict

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")
st.markdown("""<style>
.hero{background:linear-gradient(135deg,#14532d,#16a34a);padding:2rem;border-radius:16px;color:#fff;margin-bottom:1.2rem}
.hero h1{margin:0;font-size:2.2rem}.hero p{margin:.4rem 0 0;opacity:.9}
.card{border-radius:14px;padding:1.1rem 1.3rem;border:1px solid #e5e7eb;margin-bottom:.8rem}
.ok{background:#f0fdf4;border-color:#86efac}.bad{background:#fef2f2;border-color:#fca5a5}
</style>""", unsafe_allow_html=True)

INFO = {
 "scab": "Fungal disease causing dark, scabby lesions on leaves and fruit.",
 "Black_rot": "Fungal disease producing dark rotting spots on leaves and fruit.",
 "rust": "Fungal disease with orange-brown pustules on the leaf surface.",
 "Powdery_mildew": "Fungus forming a white powdery coating on leaves.",
 "spot": "Leaf spotting caused by bacteria or fungi; spreads in warm, wet conditions.",
 "blight": "Fast-spreading disease causing brown lesions and dying leaf tissue.",
 "Esca": "Fungal trunk disease causing striped, tiger-like leaf discoloration.",
 "Haunglongbing": "Bacterial citrus greening spread by psyllids; causes yellow shoots and bitter fruit.",
 "scorch": "Fungal disease leaving purple-brown blotches that dry the leaf edges.",
 "Mold": "Fungus causing yellow patches above and olive mold below the leaf.",
 "mites": "Pest damage: tiny mites cause stippling and yellowing of leaves.",
 "Virus": "Viral disease causing curling, yellowing and stunted growth.",
 "virus": "Viral disease causing mottled, distorted leaves.",
}
def describe(c):
    plant, cond = c.split("___")
    plant, cond = plant.replace("_", " "), cond.replace("_", " ").strip()
    if cond == "healthy": return plant, "Healthy", "No visible signs of disease detected."
    note = next((v for k, v in INFO.items() if k.lower() in c.lower()), "Visible disease symptoms detected.")
    return plant, cond, note

@st.cache_resource
def get_model(name): return load(name)

with st.sidebar:
    st.header("Settings")
    name = st.radio("Model", list(MODELS))
    st.caption("38 classes · 14 crops · trained on 70,295 leaf images (PlantVillage, augmented)")
    st.markdown("Built with PyTorch + Streamlit")

st.markdown('<div class="hero"><h1>🌿 Plant Disease Detection</h1>'
            '<p>Upload a leaf photo and get an instant diagnosis from a deep learning model.</p></div>',
            unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🔍 Diagnose", "📈 Training results", "ℹ️ About"])

with tab1:
    up = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up)
        c1, c2 = st.columns([1, 1.2])
        c1.image(img, width="stretch")
        with st.spinner("Analysing leaf..."):
            top = predict(get_model(name), img)
        plant, cond, note = describe(top[0][0])
        healthy = cond == "Healthy"
        c2.markdown(f'<div class="card {"ok" if healthy else "bad"}"><h3>{plant} — {cond}</h3>'
                    f'<p>{note}</p><b>Confidence: {top[0][1]:.1%}</b></div>', unsafe_allow_html=True)
        c2.caption("Top 5 predictions")
        c2.bar_chart(pd.DataFrame({"Confidence": [s for _, s in top]},
                     index=[c.replace("___", " – ").replace("_", " ") for c, _ in top]), horizontal=True)
        if top[0][1] < 0.6: st.warning("Low confidence — try a clearer, close-up photo of a single leaf.")
    else:
        st.info("Upload a JPG/PNG of a single leaf to begin.")

with tab2:
    hist = json.loads((Path(__file__).parent / "data/history.json").read_text())
    st.dataframe(pd.DataFrame([{"Model": m, "Final val accuracy": f"{h['val_acc'][-1]:.2%}",
                 "Final val loss": f"{h['val_loss'][-1]:.4f}", "Epochs": str(len(h["val_acc"]))}
                 for m, h in hist.items()] + [{"Model": "Baseline: ResNet18 features + Logistic Regression",
                 "Final val accuracy": "94.79%", "Final val loss": "–", "Epochs": "–"}]),
                 hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.subheader("Validation accuracy")
    a.line_chart(pd.DataFrame({m: h["val_acc"] for m, h in hist.items()}))
    b.subheader("Validation loss")
    b.line_chart(pd.DataFrame({m: h["val_loss"] for m, h in hist.items()}))

with tab3:
    st.markdown("""**Pipeline:** images resized to 224×224, ImageNet-normalised, augmented (flip, rotation) →
three approaches compared: a ResNet18 + Logistic Regression baseline, a ResNet18 transfer-learning model
with a custom head, and a VGG-style CNN trained from scratch.

**Dataset:** [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
— 70,295 train / 17,572 validation images, 38 classes.

*For educational use — not a substitute for an agronomist's diagnosis.*""")

