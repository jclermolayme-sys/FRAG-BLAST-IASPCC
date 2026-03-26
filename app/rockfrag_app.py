import streamlit as st
import cv2
import numpy as np
import json
import sys
import os
from pathlib import Path

# -- Configuración de Rutas (Path setup) --
# Esto permite que Python encuentre la carpeta 'core' sin importar dónde se ejecute
current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from core.segmentor import RockFragAnalyzer, RockFragVisualizer
except ImportError as e:
    st.error(f"❌ Error crítico: No se encontró el módulo core. Verifica la estructura de carpetas. Error: {e}")
    st.stop()

# -- Configuración de Página --
st.set_page_config(page_title="RockFrag AI — SPCC Cuajone", page_icon="⛏️", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    h1 { color: #00d4ff !important; font-family: 'Courier New', monospace; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #00d4ff33;
        border-radius: 8px; padding: 16px; text-align: center; margin: 4px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #00d4ff; }
    .metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

st.title("⛏️ RockFrag AI - SPCC Cuajone")
st.caption("Análisis de Fragmentación de Roca con SAM 2 (Segment Anything Model)")
st.markdown("---")

@st.cache_resource(show_spinner="Cargando modelo SAM 2...")
def get_analyzer(scale_ref: float, min_px: int, max_ratio: float) -> RockFragAnalyzer:
    return RockFragAnalyzer(
        scale_reference_cm=scale_ref,
        min_fragment_area_px=min_px,
        max_fragment_ratio=max_ratio,
        sam_model_path="sam2_t.pt" # Se descargará automáticamente si no existe
    )

# -- Barra Lateral --
with st.sidebar:
    st.header("⚙️ Parámetros")
    scale_ref_cm = st.number_input("Referencia Real (cm)", value=30.0)
    scale_mode = st.radio("Escala", ["Auto-detectar barra", "Manual (px/cm)"])
    manual_px_per_cm = st.number_input("px/cm", value=20.0) if scale_mode == "Manual (px/cm)" else None
    
    st.subheader("🔬 Filtros")
    min_frag_px = st.slider("Área mínima (px²)", 50, 2000, 300)
    max_frag_ratio = st.slider("Área máx (% de imagen)", 10, 90, 60)

# -- Área de Carga --
uploaded = st.file_uploader("Sube foto de la pila de roca", type=["jpg", "png", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Imagen original", width=600)

    if st.button("🔍 INICIAR ANÁLISIS"):
        with st.spinner("Procesando segmentación..."):
            try:
                # Usar carpeta temporal del sistema para guardar la imagen
                tmp_path = os.path.join(os.getcwd(), "temp_input.jpg")
                cv2.imwrite(tmp_path, img)

                analyzer = get_analyzer(scale_ref_cm, min_frag_px, max_frag_ratio/100)
                px_val = manual_px_per_cm if scale_mode == "Manual (px/cm)" else None
                result = analyzer.analyze(tmp_path, scale_px_per_cm=px_val)

                # -- Mostrar Métricas --
                m1, m2, m3 = st.columns(3)
                m1.markdown(f'<div class="metric-card"><div class="metric-value">{result.total_fragments}</div><div class="metric-label">Fragmentos</div></div>', unsafe_allow_html=True)
                m2.markdown(f'<div class="metric-card"><div class="metric-value">{result.p50:.1f} cm</div><div class="metric-label">P50 (Mediana)</div></div>', unsafe_allow_html=True)
                m3.markdown(f'<div class="metric-card"><div class="metric-value">{result.p80:.1f} cm</div><div class="metric-label">P80</div></div>', unsafe_allow_html=True)

                # -- Visualización --
                col1, col2 = st.columns(2)
                with col1:
                    seg_img = RockFragVisualizer.draw_segmentation(img, result)
                    st.image(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB), caption="Mapa de Segmentos")
                with col2:
                    curve_bytes = RockFragVisualizer.plot_grading_curve(result)
                    st.image(curve_bytes, caption="Curva Granulométrica")
                
                # Botón descarga
                st.download_button("Descargar JSON de Resultados", json.dumps(RockFragVisualizer.result_to_dict(result)), "resultados.json")

            except Exception as e:
                st.error(f"Error: {e}")
