import cv2
from fpdf import FPDF

# Paleta de cores global
CLUSTER_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), 
    (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 165, 255)
]

def generate_pdf_report(img_array, ai_text, stats):
    """Gera um relatorio PDF com a imagem analisada e o texto da IA."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Relatorio VisionAI Pro", ln=1, align="C")
    pdf.ln(10)
    
    # Salvar imagem temporaria
    temp_img = "temp_report.jpg"
    # Converter RGB (Streamlit) para BGR (OpenCV) antes de salvar
    cv2.imwrite(temp_img, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    pdf.image(temp_img, x=10, y=30, w=190)
    pdf.ln(110)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Metricas Detetadas:", ln=1)
    pdf.set_font("Arial", size=10)
    for k, v in stats.items():
        pdf.cell(200, 6, txt=f"{k}: {v}", ln=1)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Analise AI:", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 5, txt=ai_text)
    
    return pdf.output(dest='S').encode('latin-1', 'replace')
