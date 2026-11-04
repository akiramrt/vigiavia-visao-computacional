# API de Visão Computacional do VigiaVia (CPU, ONNX Runtime) — imagem enxuta, usuário não root.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    VIGIAVIA_PESOS=/app/modelos/vigiavia_buracos.onnx VIGIAVIA_DEVICE=cpu \
    VIGIAVIA_DB=/app/dados/vigiavia.db YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu \
    && grep -vE "^(torch|torchvision)==" requirements.txt > req-sem-torch.txt \
    && pip install --no-cache-dir -r req-sem-torch.txt

COPY src/ src/
COPY api/ api/
COPY configs/ configs/
COPY modelos/ modelos/

RUN useradd --create-home --uid 10001 vigia && mkdir -p /app/dados && chown -R vigia /app/dados
USER vigia
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
