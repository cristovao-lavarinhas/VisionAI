import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM
from ultralytics import YOLO
from PIL import Image
import torch
import cv2
import tempfile
import os
import time
import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

st.set_page_config(page_title="YOLO + Florence-2", layout="wide")
st.title("YOLO + Florence-2: Detecção e Analise Avancada")

@st.cache_resource
def load_yolo_model(model_name="yolov8n.pt"):
    return YOLO(model_name)

@st.cache_resource
def load_florence_model():
    model_id = "microsoft/Florence-2-base-ft"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype
    ).to(device).eval()
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor, device, dtype

def detect_with_yolo(image, confidence):
    yolo_model = load_yolo_model()
    results = yolo_model.predict(image, conf=confidence, verbose=False)
    return results[0]

def analyze_with_florence(image, task_prompt):
    model, processor, device, dtype = load_florence_model()
    
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
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

def analyze_image_complete(image, confidence):
    results = {}
    
    yolo_results = detect_with_yolo(image, confidence)
    boxes = yolo_results.boxes
    
    class_counts = {}
    detections_list = []
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = yolo_results.names[cls_id]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        detections_list.append({
            "classe": cls_name,
            "confianca": float(box.conf[0]),
            "bbox": box.xyxy[0].cpu().numpy().tolist()
        })
    
    results['yolo'] = {
        "total_objetos": len(boxes),
        "classes": class_counts,
        "deteccoes": detections_list,
        "imagem_anotada": yolo_results.plot()
    }
    
    st.text("Gerando descricao detalhada...")
    detailed_caption = analyze_with_florence(image, "<MORE_DETAILED_CAPTION>")
    
    st.text("Detectando objetos com Florence-2...")
    object_detection = analyze_with_florence(image, "<OD>")
    
    st.text("Analisando regioes da imagem...")
    dense_caption = analyze_with_florence(image, "<DENSE_REGION_CAPTION>")
    
    results['florence'] = {
        "descricao_detalhada": detailed_caption.get('<MORE_DETAILED_CAPTION>', ''),
        "objetos_detectados": object_detection,
        "regioes_densas": dense_caption
    }
    
    return results

def create_detection_graph(data):
    G = nx.Graph()
    
    central_node = "Analise\nImagem"
    G.add_node(central_node, node_type="root", size=5000)
    
    if 'yolo' in data:
        yolo_main = "YOLO"
        G.add_node(yolo_main, node_type="yolo_main", size=4000)
        G.add_edge(central_node, yolo_main, weight=3)
        
        for cls_name, count in data['yolo']['classes'].items():
            node_name = f"{cls_name}"
            G.add_node(node_name, node_type="yolo_class", size=2000 + count * 500, count=count)
            G.add_edge(yolo_main, node_name, weight=count)
    
    if 'florence' in data:
        florence_main = "Florence-2"
        G.add_node(florence_main, node_type="florence_main", size=4000)
        G.add_edge(central_node, florence_main, weight=3)
        
        if data['florence']['objetos_detectados']:
            obj_florence = data['florence']['objetos_detectados'].get('<OD>', {})
            if isinstance(obj_florence, dict) and 'labels' in obj_florence:
                labels_count = Counter(obj_florence['labels'])
                for label, count in list(labels_count.items())[:8]:
                    node_name = f"{label}"
                    G.add_node(node_name, node_type="florence_obj", size=2000 + count * 300, count=count)
                    G.add_edge(florence_main, node_name, weight=count)
        
        desc_node = "Descricao\nContextual"
        G.add_node(desc_node, node_type="florence_desc", size=2500)
        G.add_edge(florence_main, desc_node, weight=2)
    
    return G

def plot_network_graph(G):
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
    
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        node_type = G.nodes[node].get('node_type', 'default')
        size = G.nodes[node].get('size', 2000)
        
        if node_type == 'root':
            node_colors.append('#FF6B6B')
        elif node_type == 'yolo_main':
            node_colors.append('#4ECDC4')
        elif node_type == 'yolo_class':
            node_colors.append('#FFE66D')
        elif node_type == 'florence_main':
            node_colors.append('#95E1D3')
        elif node_type == 'florence_obj':
            node_colors.append('#A8E6CF')
        elif node_type == 'florence_desc':
            node_colors.append('#DDA0DD')
        else:
            node_colors.append('#D3D3D3')
        
        node_sizes.append(size)
    
    edges = G.edges()
    weights = [G[u][v].get('weight', 1) for u, v in edges]
    
    nx.draw_networkx_edges(
        G, pos,
        width=[w * 0.5 for w in weights],
        alpha=0.4,
        edge_color='gray',
        ax=ax
    )
    
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        linewidths=2,
        edgecolors='black',
        ax=ax
    )
    
    labels = {}
    for node in G.nodes():
        count = G.nodes[node].get('count', None)
        if count and count > 1:
            labels[node] = f"{node}\n({count})"
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=10,
        font_weight='bold',
        font_family='sans-serif',
        ax=ax
    )
    
    ax.set_title("Grafo de Deteccoes e Analises", fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    ax.margins(0.1)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=12, label='Imagem Central'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', markersize=12, label='YOLO'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFE66D', markersize=12, label='Classes YOLO'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#95E1D3', markersize=12, label='Florence-2'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#A8E6CF', markersize=12, label='Objetos Florence'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#DDA0DD', markersize=12, label='Descricao'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    return fig

def add_text_to_frame(frame, text, position=(10, 30), font_scale=0.7, thickness=2, bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
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
    
    y_offset = position[1]
    for line in lines[:3]:
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        
        cv2.rectangle(frame, 
                     (position[0] - 5, y_offset - text_height - 5),
                     (position[0] + text_width + 5, y_offset + baseline + 5),
                     bg_color, -1)
        
        cv2.putText(frame, line, (position[0], y_offset), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y_offset += text_height + 10
    
    return frame

def process_video_with_annotations(video_path, confidence, frame_skip=30, add_florence=True):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        yolo_results = detect_with_yolo(pil_image, confidence)
        annotated_frame = yolo_results.plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        boxes = yolo_results.boxes
        class_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = yolo_results.names[cls_id]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        obj_count_text = f"Objetos: {len(boxes)}"
        cv2.putText(annotated_frame, obj_count_text, (10, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        if add_florence and frame_idx % frame_skip == 0:
            try:
                florence_result = analyze_with_florence(pil_image, "<CAPTION>")
                last_caption = florence_result.get('<CAPTION>', '')
                status_text.text(f"Frame {frame_idx}/{frame_count} - Nova caption gerada")
            except Exception as e:
                status_text.text(f"Frame {frame_idx}/{frame_count} - Usando caption anterior")
        
        if add_florence and last_caption:
            annotated_frame = add_text_to_frame(annotated_frame, last_caption, (10, 30))
        
        out.write(annotated_frame)
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

st.sidebar.header("Configuracoes")

input_type = st.sidebar.radio(
    "Tipo de Entrada:",
    ["Imagem", "Video"]
)

mode = st.sidebar.radio(
    "Modo de Operacao:",
    ["YOLO + Florence-2 (Hibrido)", "Apenas YOLO", "Apenas Florence-2"]
)

if mode in ["YOLO + Florence-2 (Hibrido)", "Apenas YOLO"]:
    st.sidebar.subheader("YOLO Settings")
    yolo_model_choice = st.sidebar.selectbox(
        "Modelo YOLO:",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
    )
    confidence = st.sidebar.slider("Confianca YOLO:", 0.1, 1.0, 0.5, 0.05)

if input_type == "Video":
    st.sidebar.subheader("Video Settings")
    frame_skip = st.sidebar.slider("Gerar caption a cada N frames:", 15, 90, 30)
    add_florence_captions = st.sidebar.checkbox("Adicionar Captions Florence-2", value=True)

if input_type == "Imagem":
    uploaded_file = st.file_uploader("Upload uma imagem", type=['png', 'jpg', 'jpeg'])
else:
    uploaded_file = st.file_uploader("Upload um video", type=['mp4', 'avi', 'mov', 'mkv'])

if input_type == "Imagem" and uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Imagem Original")
        st.image(image, use_container_width=True)
    
    with col2:
        st.write("")
        st.write("")
        process_button = st.button("Processar Imagem", type="primary", use_container_width=True)
    
    if process_button:
        with st.spinner("Processando imagem completa..."):
            
            if mode == "Apenas YOLO":
                results = detect_with_yolo(image, confidence)
                boxes = results.boxes
                
                st.subheader("Resultados da Analise")
                
                col_r1, col_r2 = st.columns([2, 1])
                
                with col_r1:
                    st.write("**Imagem com Deteccoes:**")
                    annotated_img = results.plot()
                    st.image(annotated_img, use_container_width=True)
                
                with col_r2:
                    st.metric("Objetos Detectados", len(boxes))
                
                json_data = {
                    "total_objetos": len(boxes),
                    "classes": {},
                    "deteccoes": []
                }
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = results.names[cls_id]
                    json_data["classes"][cls_name] = json_data["classes"].get(cls_name, 0) + 1
                    json_data["deteccoes"].append({
                        "classe": cls_name,
                        "confianca": float(box.conf[0]),
                        "bbox": box.xyxy[0].cpu().numpy().tolist()
                    })
                
                st.write("---")
                st.subheader("Grafo de Deteccoes")
                
                graph_data = {"yolo": json_data}
                G = create_detection_graph(graph_data)
                fig = plot_network_graph(G)
                st.pyplot(fig)
                plt.close()
                
                st.write("---")
                with st.expander("Ver JSON Completo"):
                    st.json(json_data)
                
                json_str = json.dumps(json_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="yolo_analysis.json",
                    mime="application/json"
                )
            
            elif mode == "Apenas Florence-2":
                st.text("Executando analises Florence-2...")
                
                detailed_caption = analyze_with_florence(image, "<MORE_DETAILED_CAPTION>")
                object_detection = analyze_with_florence(image, "<OD>")
                dense_caption = analyze_with_florence(image, "<DENSE_REGION_CAPTION>")
                
                st.subheader("Resultados da Analise")
                
                st.write("**Descricao Detalhada da Imagem:**")
                st.markdown(f"_{detailed_caption.get('<MORE_DETAILED_CAPTION>', 'N/A')}_")
                
                florence_json = {
                    "descricao_detalhada": detailed_caption.get('<MORE_DETAILED_CAPTION>', ''),
                    "objetos_detectados": object_detection,
                    "regioes_densas": dense_caption
                }
                
                st.write("---")
                st.subheader("Grafo de Analise")
                
                graph_data = {"florence": florence_json}
                G = create_detection_graph(graph_data)
                fig = plot_network_graph(G)
                st.pyplot(fig)
                plt.close()
                
                st.write("---")
                with st.expander("Ver JSON Completo"):
                    st.json(florence_json)
                
                json_str = json.dumps(florence_json, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="florence_analysis.json",
                    mime="application/json"
                )
            
            else:
                analysis_results = analyze_image_complete(image, confidence)
                
                st.subheader("Resultados da Analise Completa")
                
                st.write("### Deteccao de Objetos (YOLO)")
                col_y1, col_y2 = st.columns([2, 1])
                
                with col_y1:
                    st.image(analysis_results['yolo']['imagem_anotada'], 
                            caption="Deteccoes YOLO", use_container_width=True)
                
                with col_y2:
                    st.metric("Objetos Detectados", analysis_results['yolo']['total_objetos'])
                
                st.write("### Analise Contextual (Florence-2)")
                st.write("**Descricao da Cena:**")
                st.markdown(f"_{analysis_results['florence']['descricao_detalhada']}_")
                
                st.write("---")
                st.subheader("Grafo de Analise Completa")
                
                combined_json = {
                    "yolo_deteccoes": {
                        "total_objetos": analysis_results['yolo']['total_objetos'],
                        "classes": analysis_results['yolo']['classes'],
                        "deteccoes": analysis_results['yolo']['deteccoes']
                    },
                    "florence_analise": analysis_results['florence']
                }
                
                graph_data = {
                    "yolo": combined_json["yolo_deteccoes"],
                    "florence": combined_json["florence_analise"]
                }
                
                G = create_detection_graph(graph_data)
                fig = plot_network_graph(G)
                st.pyplot(fig)
                plt.close()
                
                st.write("---")
                with st.expander("Ver JSON Completo"):
                    st.json(combined_json)
                
                json_str = json.dumps(combined_json, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download JSON Completo",
                    data=json_str,
                    file_name="analise_completa.json",
                    mime="application/json"
                )

elif input_type == "Video" and uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    video_path = tfile.name
    
    col_v1, col_v2 = st.columns([3, 1])
    
    with col_v1:
        st.subheader("Video Original")
        st.video(video_path)
    
    with col_v2:
        st.write("")
        st.write("")
        process_video_button = st.button("Processar Video", type="primary", use_container_width=True)
    
    if process_video_button:
        st.subheader("Processamento em Andamento")
        
        output_video_path, results_data = process_video_with_annotations(
            video_path, confidence, frame_skip, add_florence_captions
        )
        
        st.success(f"Video processado! Analisados {len(results_data)} frames")
        
        col1, col2, col3 = st.columns(3)
        
        total_objects = sum([r['objetos_detectados'] for r in results_data])
        avg_objects = total_objects / len(results_data) if results_data else 0
        
        with col1:
            st.metric("Total de Objetos", total_objects)
        with col2:
            st.metric("Media por Frame", f"{avg_objects:.1f}")
        with col3:
            st.metric("Frames Analisados", len(results_data))
        
        st.write("**Timeline de Deteccoes:**")
        for r in results_data[::max(1, len(results_data)//20)]:
            caption_info = f" - {r['caption'][:50]}..." if r['caption'] else ""
            st.write(f"Frame {r['frame']} ({r['tempo_segundos']:.1f}s): {r['objetos_detectados']} objetos{caption_info}")
        
        st.write("---")
        st.subheader("Download dos Resultados")
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            with open(output_video_path, 'rb') as f:
                video_bytes = f.read()
                st.download_button(
                    label="Download Video Anotado",
                    data=video_bytes,
                    file_name="video_anotado_yolo_florence.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
        
        with col_d2:
            json_str = json.dumps(results_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="video_analysis.json",
                mime="application/json",
                use_container_width=True
            )
        
        with st.expander("Ver JSON Completo"):
            st.json(results_data)
        
        try:
            time.sleep(0.5)
            if os.path.exists(output_video_path):
                os.unlink(output_video_path)
        except:
            pass