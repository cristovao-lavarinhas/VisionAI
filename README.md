# VisionAI

Deteção de objetos e leitura de cenas, sobre a mesma imagem.

**[site do VisionAI](https://cristovao-lavarinhas.github.io/VisionAI/)**

Uma caixa diz **onde** está a coisa. Uma legenda diz **o que ali se passa.** O
VisionAI corre os dois em cima do mesmo ficheiro — **YOLOv8** para as caixas,
classes e confianças, **Florence-2** para a descrição em linguagem natural — e
junta as duas leituras num só JSON. Funciona sobre imagem e sobre vídeo, e os
modelos correm localmente.

## Stack

Streamlit · Ultralytics YOLOv8 · Florence-2 (`microsoft/Florence-2-base-ft`) ·
Transformers · PyTorch · OpenCV · Pillow

## Arranque

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre em http://localhost:8501. Os pesos do YOLOv8 nano (`yolov8n.pt`) estão
versionados no repositório; o Florence-2 é descarregado do Hugging Face na
primeira utilização — cerca de 0,5 GB, uma vez só.

O Florence-2 carrega com `trust_remote_code=True`, porque a arquitetura vive no
próprio repositório do modelo e não no `transformers` — daí o `timm` e o
`einops` nas dependências. Com CUDA disponível corre em `float16`; sem ela, cai
para `float32` em CPU, que funciona mas é bastante mais lento.

O `.devcontainer/` está configurado para Codespaces: Python 3.11, instala o
`requirements.txt` e arranca o Streamlit na porta 8501.

## Modos

Escolhidos na barra lateral, e válidos para imagem ou vídeo.

| Modo | O que corre |
| --- | --- |
| YOLO + Florence-2 | Deteção, descrição detalhada, deteção do Florence-2 e legendagem densa por região |
| Apenas YOLO | Só caixas, classes e confianças — rápido |
| Apenas Florence-2 | Só a leitura da cena, sem caixas |

O limiar de confiança do YOLO vai de 0,1 a 1,0 (0,5 por omissão).

## O que sai

| Entrada | Resultado |
| --- | --- |
| Imagem | Imagem anotada, contagem de objetos, descrição da cena, e JSON com cada deteção — classe, confiança e bounding box |
| Vídeo | MP4 anotado com as caixas, a legenda e a contagem gravadas na imagem, mais um JSON por frame com tempo, contagem, classes e legenda |

No vídeo aparecem também três métricas agregadas: total de objetos, média por
frame e número de frames analisados.

## Arquitetura

```
app.py            carregamento dos modelos, analise de imagem e de video, UI
yolov8n.pt        pesos do YOLOv8 nano, versionados no repo
requirements.txt
docs/index.html   landing page do projeto
.devcontainer/    Codespaces — Python 3.11, Streamlit na porta 8501
```

### Dois modelos, dois papéis

O YOLO devolve geometria: para cada objeto, uma classe, uma confiança e quatro
coordenadas. O Florence-2 devolve texto, e é chamado com uma tarefa diferente
conforme o que se quer:

| Tarefa | Para quê |
| --- | --- |
| `<MORE_DETAILED_CAPTION>` | Descrição longa da cena inteira |
| `<OD>` | Deteção de objetos pelo lado do Florence-2, para comparar com a do YOLO |
| `<DENSE_REGION_CAPTION>` | Uma legenda por região, não uma por imagem |
| `<CAPTION>` | Versão curta, usada nos frames de vídeo |

As duas leituras não se substituem — ficam lado a lado no JSON final, sob
`yolo_deteccoes` e `florence_analise`.

### Vídeo: o custo manda no ritmo

Os dois modelos têm custos muito diferentes, e o processamento de vídeo é
desenhado à volta disso. O **YOLO corre em todos os frames** — é barato e a
contagem tem de acompanhar a ação. O **Florence-2 corre a cada N frames**
(ajustável entre 15 e 90, 30 por omissão); nos frames intermédios a última
legenda é reutilizada e gravada na imagem.

Se uma chamada ao Florence-2 falhar, o frame não fica sem legenda: mantém a
anterior e o processamento segue. A legenda é quebrada por palavras à largura do
frame e cortada às três linhas, sobre um retângulo opaco, para nunca sair fora
da imagem.

### Cache de modelos

Ambos os carregamentos estão sob `@st.cache_resource`, por isso os modelos são
lidos uma vez e reutilizados entre execuções — sem isto, cada clique no botão
recarregaria o Florence-2 do disco.

## Limitações conhecidas

- O seletor de modelo YOLO na barra lateral (`yolov8n/s/m/l`) ainda não está
  ligado ao carregamento — corre sempre o `yolov8n.pt`.
- Em vídeo, o modo **Apenas Florence-2** falha: o limiar de confiança só é
  definido nos modos que usam YOLO, e o processamento de vídeo pede-o sempre.

## Landing page

`docs/index.html` é um ficheiro único, estático, sem framework nem build step,
servido pelo GitHub Pages a partir da pasta `/docs` do `main`.
