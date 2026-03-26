import streamlit as st
import cv2
import numpy as np
import sys
from pathlib import Path

# Configuración de rutas para GitHub
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path: sys.path.insert(0, str(root))

from core.segmentor import RockFragAnalyzer, RockFragVisualizer

st.set_page_config(page_title="RockFrag Cuajone", layout="centered")

# Estilo CSS para móvil
st.markdown("""<style>
    .stButton>button { width: 100%; border-radius: 10px; background: #00d4ff; color: black; font-weight: bold; }
    .metric-box { background: #16213e; padding: 15px; border-radius: 10px; text-align: center; }
</style>""", unsafe_allow_html=True)

st.title("⛏️ RockFrag AI - SPCC")

@st.cache_resource
def load_engine():
    return RockFragAnalyzer()

engine = load_engine()

# Entrada de imagen: Cámara por defecto para el celular
img_file = st.camera_input("Tomar foto de fragmentación")

if img_file:
    bytes_data = img_file.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    if st.button("ANALIZAR AHORA"):
        with st.spinner("Procesando..."):
            res = engine.analyze(img)
            
            # Métricas en columnas
            col1, col2 = st.columns(2)
            col1.metric("Fragmentos", res.total_fragments)
            col2.metric("P80 (cm)", f"{res.p80:.1f}")
            
            # Imagen Resultante
            st.image(cv2.cvtColor(RockFragVisualizer.draw(img, res), cv2.COLOR_BGR2RGB), caption="Segmentación")
