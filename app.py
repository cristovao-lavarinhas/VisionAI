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
    
    # Botão para abrir modal
    if st.button("Configurar YOLO", use_container_width=True):
        st.session_state['show_config_modal'] = True
    
    # Mostrar valores atuais
    st.caption(f"**Confiança:** {st.session_state.get('conf_thresh', 0.4)}")
    st.caption(f"**Agrupamento:** {st.session_state.get('eps_val', 100)}")
    
    # Mostrar mensagem de sucesso/erro se existir
    if 'config_message' in st.session_state:
        msg_type = st.session_state['config_message']['type']
        msg_text = st.session_state['config_message']['text']
        
        if msg_type == 'success':
            st.success(msg_text)
        elif msg_type == 'error':
            st.error(msg_text)
        
        # Limpar mensagem após mostrar
        if st.button("✖ Fechar", key="close_msg"):
            del st.session_state['config_message']
            st.rerun()

# =========================
# MODAL DE CONFIGURAÇÃO
# =========================
@st.dialog("Configurações YOLO")
def config_modal():
    st.markdown("### Parâmetros de Detecção")
    
    # Sliders dentro da modal
    conf_thresh = st.slider(
        "Confiança Mínima",
        0.1, 1.0,
        st.session_state.get('conf_thresh', 0.4),
        0.05,
        help="Threshold mínimo para considerar uma detecção válida (0.1 - 1.0)"
    )
    
    eps_val = st.slider(
        "Distância de Agrupamento (DBSCAN)",
        30, 300,
        st.session_state.get('eps_val', 100),
        10,
        help="Distância máxima entre pontos para formar um cluster (30 - 300)"
    )
    
    # Mostrar preview das mudanças
    if conf_thresh != st.session_state.get('conf_thresh', 0.4) or eps_val != st.session_state.get('eps_val', 100):
        st.info("Mudanças pendentes. Clique em 'Aplicar' para confirmar.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Aplicar", use_container_width=True):
            try:
                # Validações
                errors = []
                
                # Validar confiança
                if not (0.1 <= conf_thresh <= 1.0):
                    errors.append("Confiança deve estar entre 0.1 e 1.0")
                
                # Validar distância de agrupamento
                if not (30 <= eps_val <= 300):
                    errors.append("Distância de agrupamento deve estar entre 30 e 300")
                
                # Se houver erros, mostrar
                if errors:
                    error_msg = "Erros de validação:\n" + "\n".join(f"• {e}" for e in errors)
                    st.session_state['config_message'] = {
                        'type': 'error',
                        'text': error_msg
                    }
                    st.session_state['show_config_modal'] = False
                    st.rerun()
                    return
                
                # Testar se os valores funcionam (simulação)
                try:
                    # Aqui podes adicionar um teste real se necessário
                    # Por exemplo: testar com uma imagem pequena
                    test_passed = True
                    
                    if test_passed:
                        # Guardar valores
                        old_conf = st.session_state.get('conf_thresh', 0.4)
                        old_eps = st.session_state.get('eps_val', 100)
                        
                        st.session_state['conf_thresh'] = conf_thresh
                        st.session_state['eps_val'] = eps_val
                        
                        # Mensagem de sucesso
                        st.session_state['config_message'] = {
                            'type': 'success',
                            'text': f"Configurações aplicadas com sucesso!\n\n" +
                                   f"Confiança: {old_conf} → {conf_thresh}\n" +
                                   f"Agrupamento: {old_eps} → {eps_val}"
                        }
                        
                        st.session_state['show_config_modal'] = False
                        st.rerun()
                    
                except Exception as e:
                    # Erro durante o teste
                    st.session_state['config_message'] = {
                        'type': 'error',
                        'text': f"Erro ao aplicar configurações:\n\n{type(e).__name__}: {str(e)}"
                    }
                    st.session_state['show_config_modal'] = False
                    st.rerun()
            
            except Exception as e:
                # Erro inesperado
                st.session_state['config_message'] = {
                    'type': 'error',
                    'text': f"Erro crítico:\n\n{type(e).__name__}: {str(e)}"
                }
                st.session_state['show_config_modal'] = False
                st.rerun()
    
    with col2:
        if st.button("Cancelar", use_container_width=True):
            st.session_state['show_config_modal'] = False
            st.rerun()

# Mostrar modal se a flag estiver ativa
if st.session_state.get('show_config_modal', False):
    config_modal()

# Obter valores de configuração
conf_thresh = st.session_state.get('conf_thresh', 0.4)
eps_val = st.session_state.get('eps_val', 100)

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
        
        try:
            with st.spinner("A processar..."):
                p_frame, classes, labels = engine.process_frame(frame_rgb, conf_thresh, eps_val)
            
            # Layout Lado a Lado
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Original")
                st.image(frame_rgb, width='stretch')
            with col2:
                st.markdown("### YOLOv8")
                st.image(p_frame, width='stretch')
                
            st.divider()
            col_metrics, col_btn = st.columns([2, 1])
            
            with col_metrics:
                c_a, c_b = st.columns(2)
                c_a.metric("Total Objetos", len(classes))
                c_b.metric("Grupos", len(set(labels)) if len(labels) > 0 else 0)
                
                # Mostrar estrutura de dados simples
                clusters = {}
                for c, l in zip(classes, labels): 
                    clusters.setdefault(str(l), []).append(c)
                with st.expander("Ver Detalhes dos Grupos"):
                    st.json(clusters)

            with col_btn:
                # Relatorio PDF simples
                stats = {"Objetos": len(classes), "Grupos": len(set(labels))}
                pdf = generate_pdf_report(p_frame, "Relatorio gerado automaticamente pelo VisionAI Pro.", stats)
                st.download_button("Baixar Relatório PDF", pdf, "relatorio.pdf", "application/pdf")
        
        except Exception as e:
            st.error(f"Erro ao processar imagem:\n\n{type(e).__name__}: {str(e)}")
            st.info("Tente ajustar os parâmetros YOLO na sidebar.")

# === TAB 2: VIDEO ===
with tab2:
    vid_file = st.file_uploader("Carregar MP4", type=['mp4'])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(vid_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        if st.button("Iniciar Processamento"):
            try:
                prog = st.progress(0)
                status = st.empty()
                
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                out = cv2.VideoWriter("video_out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                
                cnt = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
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
            
            except Exception as e:
                st.error(f"Erro ao processar vídeo:\n\n{type(e).__name__}: {str(e)}")
                st.info("Tente ajustar os parâmetros YOLO ou verificar o formato do vídeo.")
