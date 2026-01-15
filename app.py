import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
import tempfile
import os
import time

# Configuracao da pagina
st.set_page_config(page_title="YOLO + Florence-2", layout="wide")
st.title("YOLO + Florence-2: Detecção e Analise Avancada")

# Cache dos modelos
@st.cache_resource
def load_yolo_model(model_name="yolov8n.pt"):
    """Carrega modelo YOLO"""
    return YOLO(model_name)

@st.cache_resource
def load_florence_model():
    """Carrega modelo Florence-2"""
    model_id = "microsoft/Florence-2-base-ft"
    
    # Verifica se tem GPU disponivel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype
    ).to(device).eval()
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor, device, dtype

# Funcao para deteccao YOLO
def detect_with_yolo(image, confidence):
    """Executa deteccao YOLO e retorna resultados"""
    yolo_model = load_yolo_model()
    results = yolo_model.predict(image, conf=confidence, verbose=False)
    return results[0]

# Funcao para analise Florence-2 com contexto detalhado
def analyze_with_florence(image, task_prompt, text_input=None):
    """Executa analise Florence-2"""
    model, processor, device, dtype = load_florence_model()
    
    prompt = task_prompt + (text_input if text_input else "")
    
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # Move inputs para o device correto e converte para o tipo correto
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Converte pixel_values para o mesmo dtype do modelo
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            early_stopping=False
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    return parsed_answer

# Funcao para gerar descricao ultra detalhada
def get_detailed_scene_description(image):
    """Gera descricao detalhada da cena incluindo objetos, interacoes e contexto"""
    
    # Primeira passagem: descricao geral muito detalhada
    detailed_caption = analyze_with_florence(image, "<MORE_DETAILED_CAPTION>")
    
    # Segunda passagem: deteccao de objetos
    object_detection = analyze_with_florence(image, "<OD>")
    
    # Terceira passagem: legendas de regioes densas
    dense_caption = analyze_with_florence(image, "<DENSE_REGION_CAPTION>")
    
    # Quarta passagem: OCR se houver texto
    ocr_result = analyze_with_florence(image, "<OCR>")
    
    result = {
        "descricao_geral": detailed_caption.get('<MORE_DETAILED_CAPTION>', ''),
        "objetos_detectados": object_detection,
        "descricao_regioes": dense_caption,
        "texto_extraido": ocr_result
    }
    
    return result

# Funcao para adicionar texto formatado ao frame
def add_text_to_frame(frame, text, position=(10, 30), font_scale=0.7, thickness=2, bg_color=(0, 0, 0)):
    """Adiciona texto com fundo ao frame do video"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Divide texto em linhas se for muito longo
    max_width = frame.shape[1] - 20
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        
        if text_width < max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Desenha cada linha
    y_offset = position[1]
    for line in lines[:3]:  # Limita a 3 linhas
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        
        # Desenha retangulo de fundo
        cv2.rectangle(frame, 
                     (position[0] - 5, y_offset - text_height - 5),
                     (position[0] + text_width + 5, y_offset + baseline + 5),
                     bg_color, -1)
        
        # Desenha texto
        cv2.putText(frame, line, (position[0], y_offset), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y_offset += text_height + 10
    
    return frame

# Funcao para processar video com anotacoes
def process_video_with_annotations(video_path, confidence, frame_skip=30, add_florence=True):
    """Processa video frame a frame e gera video anotado"""
    cap = cv2.VideoCapture(video_path)
    
    # Propriedades do video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Cria arquivo de saida temporario
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_idx = 0
    last_caption = ""
    results_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converte BGR para RGB para processamento
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Deteccao YOLO
        yolo_results = detect_with_yolo(pil_image, confidence)
        
        # Desenha deteccoes YOLO no frame
        annotated_frame = yolo_results.plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        # Contagem de objetos
        boxes = yolo_results.boxes
        class_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = yolo_results.names[cls_id]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        # Adiciona contador de objetos
        obj_count_text = f"Objetos: {len(boxes)}"
        cv2.putText(annotated_frame, obj_count_text, (10, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Gera caption com Florence-2 periodicamente
        if add_florence and frame_idx % frame_skip == 0:
            try:
                florence_result = analyze_with_florence(pil_image, "<CAPTION>")
                last_caption = florence_result.get('<CAPTION>', '')
                status_text.text(f"Frame {frame_idx}/{frame_count} - Nova caption gerada")
            except Exception as e:
                status_text.text(f"Frame {frame_idx}/{frame_count} - Usando caption anterior")
        
        # Adiciona caption ao frame
        if add_florence and last_caption:
            annotated_frame = add_text_to_frame(annotated_frame, last_caption, (10, 30))
        
        # Escreve frame anotado
        out.write(annotated_frame)
        
        # Atualiza progresso
        progress_bar.progress(min(frame_idx / frame_count, 1.0))
        if frame_idx % 10 == 0:
            status_text.text(f"Processando frame {frame_idx}/{frame_count}")
        
        results_data.append({
            "frame": frame_idx,
            "tempo_segundos": frame_idx / fps,
            "objetos_detectados": len(boxes),
            "contagem_classes": class_counts,
            "caption": last_caption if frame_idx % frame_skip == 0 else ""
        })
        
        frame_idx += 1
    
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    return output_path, results_data

# Funcao para criar ROIs a partir de deteccoes YOLO
def extract_roi_from_yolo(image, box):
    """Extrai regiao de interesse da deteccao YOLO"""
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    roi = image.crop((x1, y1, x2, y2))
    return roi, (x1, y1, x2, y2)

# Sidebar - Configuracoes
st.sidebar.header("Configuracoes")

# Selecao de tipo de entrada
input_type = st.sidebar.radio(
    "Tipo de Entrada:",
    ["Imagem", "Video"]
)

# Selecao de modo
mode = st.sidebar.radio(
    "Modo de Operacao:",
    ["YOLO + Florence-2 (Hibrido)", "Apenas YOLO", "Apenas Florence-2"]
)

# Configuracoes YOLO
if mode in ["YOLO + Florence-2 (Hibrido)", "Apenas YOLO"]:
    st.sidebar.subheader("YOLO Settings")
    yolo_model_choice = st.sidebar.selectbox(
        "Modelo YOLO:",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
    )
    confidence = st.sidebar.slider("Confianca YOLO:", 0.1, 1.0, 0.5, 0.05)

# Configuracoes Florence-2
if mode in ["YOLO + Florence-2 (Hibrido)", "Apenas Florence-2"]:
    st.sidebar.subheader("Florence-2 Settings")
    detailed_analysis = st.sidebar.checkbox("Analise Ultra Detalhada", value=True, 
                                           help="Inclui interacoes, contexto e multiplas camadas de analise")

# Configuracoes de video
if input_type == "Video":
    st.sidebar.subheader("Video Settings")
    frame_skip = st.sidebar.slider("Gerar caption a cada N frames:", 15, 90, 30)
    add_florence_captions = st.sidebar.checkbox("Adicionar Captions Florence-2", value=True)

# Upload de arquivo
if input_type == "Imagem":
    uploaded_file = st.file_uploader("Upload uma imagem", type=['png', 'jpg', 'jpeg'])
else:
    uploaded_file = st.file_uploader("Upload um video", type=['mp4', 'avi', 'mov', 'mkv'])

# Processamento de IMAGEM
if input_type == "Imagem" and uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(image, use_container_width=True)
    
    # Botao de processamento
    if st.button("Processar", type="primary"):
        with col2:
            st.subheader("Resultados")
            
            # Modo: Apenas YOLO
            if mode == "Apenas YOLO":
                with st.spinner("Detectando com YOLO..."):
                    results = detect_with_yolo(image, confidence)
                    
                    # Imagem anotada
                    annotated_img = results.plot()
                    st.image(annotated_img, caption="Deteccoes YOLO", use_container_width=True)
                    
                    # Estatisticas
                    boxes = results.boxes
                    st.metric("Objetos Detectados", len(boxes))
                    
                    # Contagem por classe
                    if len(boxes) > 0:
                        class_counts = {}
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            cls_name = results.names[cls_id]
                            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                        
                        st.write("**Contagem por Classe:**")
                        for cls_name, count in class_counts.items():
                            st.write(f"- {cls_name}: {count}")
                        
                        # Opcao para ver JSON
                        if st.checkbox("Ver JSON completo", key="yolo_json"):
                            json_data = {
                                "total_objetos": len(boxes),
                                "classes": class_counts,
                                "deteccoes": []
                            }
                            for box in boxes:
                                json_data["deteccoes"].append({
                                    "classe": results.names[int(box.cls[0])],
                                    "confianca": float(box.conf[0]),
                                    "bbox": box.xyxy[0].cpu().numpy().tolist()
                                })
                            st.json(json_data)
            
            # Modo: Apenas Florence-2
            elif mode == "Apenas Florence-2":
                with st.spinner("Analisando com Florence-2..."):
                    if detailed_analysis:
                        result = get_detailed_scene_description(image)
                        
                        st.write("**Descricao Geral da Cena:**")
                        st.markdown(f"_{result['descricao_geral']}_")
                        
                        st.write("**Descricao de Regioes e Interacoes:**")
                        dense_regions = result['descricao_regioes'].get('<DENSE_REGION_CAPTION>', {})
                        if isinstance(dense_regions, dict) and 'labels' in dense_regions:
                            for label in dense_regions['labels']:
                                st.write(f"- {label}")
                        
                        if result['texto_extraido'].get('<OCR>', ''):
                            st.write("**Texto Encontrado na Imagem:**")
                            st.info(result['texto_extraido']['<OCR>'])
                        
                        # Opcao para ver JSON
                        if st.checkbox("Ver JSON completo", key="florence_json"):
                            st.json(result)
                    else:
                        result = analyze_with_florence(image, "<MORE_DETAILED_CAPTION>")
                        st.write("**Resultado Florence-2:**")
                        for key, value in result.items():
                            if isinstance(value, str):
                                st.markdown(f"**{key}:**")
                                st.write(value)
                            else:
                                st.json({key: value})
            
            # Modo: Hibrido YOLO + Florence-2
            else:
                with st.spinner("Detectando com YOLO..."):
                    yolo_results = detect_with_yolo(image, confidence)
                    boxes = yolo_results.boxes
                    
                    st.write(f"**YOLO detectou {len(boxes)} objetos**")
                    
                    # Mostra imagem com deteccoes YOLO
                    annotated_img = yolo_results.plot()
                    st.image(annotated_img, caption="Deteccoes YOLO", use_container_width=True)
                
                # Analise Florence-2 da imagem completa
                with st.spinner("Analisando contexto com Florence-2..."):
                    if detailed_analysis:
                        florence_result = get_detailed_scene_description(image)
                        
                        st.write("**Descricao Contextual Completa:**")
                        st.markdown(f"_{florence_result['descricao_geral']}_")
                        
                        st.write("**Analise de Interacoes e Elementos:**")
                        dense_regions = florence_result['descricao_regioes'].get('<DENSE_REGION_CAPTION>', {})
                        if isinstance(dense_regions, dict) and 'labels' in dense_regions:
                            for idx, label in enumerate(dense_regions['labels'][:10]):
                                st.write(f"{idx+1}. {label}")
                        
                        if florence_result['texto_extraido'].get('<OCR>', ''):
                            st.write("**Texto Detectado:**")
                            st.info(florence_result['texto_extraido']['<OCR>'])
                        
                        # JSON completo
                        if st.checkbox("Ver JSON completo da analise", key="hybrid_json"):
                            combined_json = {
                                "yolo_deteccoes": {
                                    "total_objetos": len(boxes),
                                    "classes": {}
                                },
                                "florence_analise": florence_result
                            }
                            
                            for box in boxes:
                                cls_name = yolo_results.names[int(box.cls[0])]
                                if cls_name not in combined_json["yolo_deteccoes"]["classes"]:
                                    combined_json["yolo_deteccoes"]["classes"][cls_name] = 0
                                combined_json["yolo_deteccoes"]["classes"][cls_name] += 1
                            
                            st.json(combined_json)
                    else:
                        florence_result = analyze_with_florence(image, "<MORE_DETAILED_CAPTION>")
                        description = florence_result.get('<MORE_DETAILED_CAPTION>', 'N/A')
                        st.write("**Descricao Detalhada da Imagem:**")
                        st.markdown(f"_{description}_")
                
                # Analise detalhada de cada objeto detectado
                if len(boxes) > 0 and st.checkbox("Analisar cada objeto com Florence-2", value=False):
                    st.write("---")
                    st.write("**Analise Individual dos Objetos:**")
                    
                    for idx, box in enumerate(boxes[:5]):
                        cls_id = int(box.cls[0])
                        cls_name = yolo_results.names[cls_id]
                        conf = float(box.conf[0])
                        
                        with st.expander(f"Objeto {idx+1}: {cls_name} ({conf:.2%})"):
                            roi, coords = extract_roi_from_yolo(image, box)
                            
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.image(roi, caption=f"ROI: {cls_name}", use_container_width=True)
                            
                            with col_b:
                                florence_roi = analyze_with_florence(roi, "<DETAILED_CAPTION>")
                                desc = florence_roi.get('<DETAILED_CAPTION>', 'N/A')
                                st.write(f"**Descricao:** {desc}")
                                st.write(f"**Coordenadas:** {coords}")

# Processamento de VIDEO
elif input_type == "Video" and uploaded_file:
    # Salva video temporariamente
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()  # Fecha o arquivo antes de usar
    video_path = tfile.name
    
    st.subheader("Video Carregado")
    st.video(video_path)
    
    if st.button("Processar Video", type="primary"):
        st.subheader("Processamento em Andamento")
        
        # Processa video com anotacoes
        output_video_path, results_data = process_video_with_annotations(
            video_path, confidence, frame_skip, add_florence_captions
        )
        
        st.success(f"Video processado! Analisados {len(results_data)} frames")
        
        # Mostra video processado
        st.subheader("Video Processado com Anotacoes")
        st.video(output_video_path)
        
        # Botao para download do video
        with open(output_video_path, 'rb') as f:
            video_bytes = f.read()
            st.download_button(
                label="Download Video Anotado",
                data=video_bytes,
                file_name="video_anotado_yolo_florence.mp4",
                mime="video/mp4"
            )
        
        # Estatisticas gerais
        col1, col2, col3 = st.columns(3)
        
        total_objects = sum([r['objetos_detectados'] for r in results_data])
        avg_objects = total_objects / len(results_data) if results_data else 0
        
        with col1:
            st.metric("Total de Objetos Detectados", total_objects)
        with col2:
            st.metric("Media por Frame", f"{avg_objects:.1f}")
        with col3:
            st.metric("Frames Analisados", len(results_data))
        
        # Timeline de deteccoes
        st.write("**Timeline de Deteccoes:**")
        for r in results_data[::max(1, len(results_data)//20)]:  # Mostra amostra de 20
            caption_info = f" - Caption: {r['caption'][:50]}..." if r['caption'] else ""
            st.write(f"Frame {r['frame']} ({r['tempo_segundos']:.1f}s): {r['objetos_detectados']} objetos - {r['contagem_classes']}{caption_info}")
        
        # JSON completo
        if st.checkbox("Ver JSON completo do video", key="video_json"):
            st.json(results_data)
        
        # Exportar resultados
        if st.button("Exportar Resultados JSON"):
            import json
            json_str = json.dumps(results_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="video_analysis_results.json",
                mime="application/json"
            )
        
        # Limpa video processado depois de mostrar
        time.sleep(1)
        try:
            if os.path.exists(output_video_path):
                os.unlink(output_video_path)
        except:
            pass
    
    # Limpa arquivo temporario original quando sair da pagina
    # Usa session state para controlar a limpeza
    if 'temp_video_path' not in st.session_state:
        st.session_state.temp_video_path = video_path

