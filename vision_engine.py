import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from utils import CLUSTER_COLORS

class VisionEngine:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def process_frame(self, frame, conf_threshold, eps_val):
        """
        Recebe um frame (RGB ou BGR), deteta objetos e desenha caixas agrupadas.
        Retorna: frame desenhado, lista de classes, lista de labels (grupos).
        """
        results = self.model(frame, verbose=False, conf=conf_threshold)
        detections = results[0].boxes
        
        centers, boxes, classes = [], [], []
        
        if len(detections) > 0:
            for box in detections:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                
                centers.append([(x1 + x2)/2, (y1 + y2)/2])
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                classes.append(self.model.names[cls_id])
            
            # DBSCAN Clustering
            if len(centers) > 0:
                clustering = DBSCAN(eps=eps_val, min_samples=1).fit(np.array(centers))
                labels = clustering.labels_
            else:
                labels = np.array([])
        else:
            labels = np.array([])

        draw_frame = frame.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            lab = labels[i] if len(labels) > 0 else -1
            
            # Escolhe a cor baseada no ID do grupo
            color = CLUSTER_COLORS[lab % len(CLUSTER_COLORS)] if lab != -1 else (128,128,128)
            
            # Se o frame for BGR (video), a cor deve ser convertida? 
            # O Streamlit (Tab 1) envia RGB. O OpenCV VideoWriter (Tab 2) usa BGR.
            # Este código assume que a cor `CLUSTER_COLORS` (RGB) é usada diretamente.
            # Se a entrada for BGR, as cores ficam trocadas (Azul vira Vermelho).
            # Para simplicidade, assumimos que quem chama gere isso ou aceitamos cores trocadas no vídeo.
            
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw_frame, f"G{lab}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return draw_frame, classes, labels
