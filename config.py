# config.py
from pathlib import Path

# --- Rutas Base ---
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

# --- Nombres de Modelos ---
DETECTION_MODEL_NAME = "Yolo11x-exp2-best2.pt"
SEGMENTATION_MODEL_NAME = "Yolov11x-C3CA-best.pt"

# --- Rutas de Modelos Completas ---
DETECTION_MODEL_PATH = MODELS_DIR / DETECTION_MODEL_NAME
SEGMENTATION_MODEL_PATH = MODELS_DIR / SEGMENTATION_MODEL_NAME

# --- Ruta de Salida Final ---
# El reporte CSV es el único archivo de salida que es global para todo el proceso.
FINAL_REPORT_PATH = OUTPUT_DIR / "final_report.csv"

# --- Parámetros del Pipeline ---
# Confianza mínima para la detección de estomas en la Etapa 1
DETECTION_CONFIDENCE = 0.81 
# Confianza mínima para la segmentación en la Etapa 3
SEGMENTATION_CONFIDENCE = 0.25 # Te sugiero mantenerlo bajo para evitar las advertencias

# Asegurarse de que las carpetas base existan
OUTPUT_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)