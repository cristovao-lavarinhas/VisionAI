import streamlit as st
import cv2
import tempfile
import numpy as np
from vision_engine import VisionEngine
from utils import generate_pdf_report

# =========================
# CONFIGURACAO
# =========================
st.set_page_config(page_title="Project III", layout="wide")

# Inicializar Motor (Cache)
@st.cache_resource
def get_engine():
    return VisionEngine()

engine = get_engine()

st.title("")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Configuração")
    st.subheader("Parâmetros YOLO")
    conf_thresh = st.slider("Confiança Mínima", 0.1, 1.0, 0.4, 0.05)
    eps_val = st.slider("Distância Agrupamento", 30, 300, 100, 10)

# =========================
# ABAS PRINCIPAIS
# =========================
tab1, tab2 = st.tabs(["Analise de Imagem", "Processamento de Video"])

# === TAB 1: IMAGEM ===
with tab1:
    img_file = st.file_uploader("Carregar Imagem", type=['jpg', 'png', 'jpeg'])
    
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        with st.spinner("A processar..."):
            p_frame, classes, labels = engine.process_frame(frame_rgb, conf_thresh, eps_val)
        
        # Layout Lado a Lado
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Original")
            st.image(frame_rgb, use_container_width=True)
        with col2:
            st.markdown("### Analise VisionAI")
            st.image(p_frame, use_container_width=True)
            
        st.divider()
        col_metrics, col_btn = st.columns([2, 1])
        
        with col_metrics:
            c_a, c_b = st.columns(2)
            c_a.metric("Total Objetos", len(classes))
            c_b.metric("Grupos", len(set(labels)) if len(labels) > 0 else 0)
            
            # Mostrar estrutura de dados simples
            clusters = {}
            for c, l in zip(classes, labels): clusters.setdefault(str(l), []).append(c)
            with st.expander("Ver Detalhes dos Grupos"):
                st.json(clusters)

        with col_btn:
            # Relatorio PDF simples (sem texto de IA)
            stats = {"Objetos": len(classes), "Grupos": len(set(labels))}
            pdf = generate_pdf_report(p_frame, "Relatorio gerado automaticamente pelo VisionAI Pro.", stats)
            st.download_button("Baixar Relatório PDF", pdf, "relatorio.pdf", "application/pdf")

# === TAB 2: VIDEO ===
with tab2:
    vid_file = st.file_uploader("Carregar MP4", type=['mp4'])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(vid_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        if st.button("Iniciar Processamento"):
            prog = st.progress(0)
            status = st.empty()
            
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            out = cv2.VideoWriter("video_out.mp4", cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
            
            cnt = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Processamento
                p_frame, _, _ = engine.process_frame(frame, conf_thresh, eps_val)
                out.write(p_frame)
                
                cnt += 1
                if cnt % 10 == 0: 
                    prog.progress(min(cnt/total, 1.0))
                    status.text(f"Frame {cnt}/{total}")
            
            cap.release()
            out.release()
            prog.progress(100)
            status.success("Concluido!")
            
            with open("video_out.mp4", "rb") as f:
                st.download_button("Baixar Video", f.read(), "video.mp4")
