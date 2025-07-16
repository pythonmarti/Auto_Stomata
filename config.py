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

# --- Rutas de Salida de cada Etapa ---
# Nota: Ultralytics crea su propia carpeta 'runs/detect/predict', 
# pero centralizamos nuestras salidas personalizadas aquí.
CROPPED_OUTPUT_DIR = OUTPUT_DIR / "2_cropped_stomata"
FINAL_REPORT_PATH = OUTPUT_DIR / "final_report.csv"

# --- Parámetros del Pipeline ---
# Confianza mínima para la detección de estomas en la Etapa 1
DETECTION_CONFIDENCE = 0.85 
# Confianza mínima para la segmentación en la Etapa 3
SEGMENTATION_CONFIDENCE = 0.25

# Asegurarse de que las carpetas de salida existan
OUTPUT_DIR.mkdir(exist_ok=True)
CROPPED_OUTPUT_DIR.mkdir(exist_ok=True)