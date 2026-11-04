"""VigiaVia — componente de Visão Computacional (detecção de buracos em vias públicas)."""
import os

# Segurança: o Ultralytics tenta instalar pacotes via pip em tempo de execução quando falta algo
# (ex.: onnxruntime-gpu). Em produção isso é indesejável — dependências vêm só do requirements.txt.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

__version__ = "1.0.0"
