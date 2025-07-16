# pipeline_stages.py
import cv2
import torch
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import numpy as np

# Importar configuración
import config

def stage_1_detect_stomata(input_dir: Path, model_path: Path, confidence: float):
    """
    Etapa 1: Detecta estomas en las imágenes de entrada usando un modelo YOLO.

    Args:
        input_dir (Path): Carpeta con las imágenes de entrada.
        model_path (Path): Ruta al modelo de detección .pt.
        confidence (float): Umbral de confianza para las detecciones.

    Returns:
        dict: Un diccionario donde las claves son los IDs de imagen y los valores
              son listas de tuplas (imagen_original, cajas_delimitadoras).
    """
    print("--- Iniciando Etapa 1: Detección de Estomas ---")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo de detección no encontrado en: {model_path}")

    model = YOLO(model_path)
    detection_results = {}

    # Validar en todas las imágenes del directorio de entrada
    # Usamos `stream=True` para procesar imágenes grandes eficientemente
    results_generator = model.predict(source=str(input_dir), conf=confidence, stream=True)

    for result in results_generator:
        image_path = Path(result.path)
        image_id = image_path.stem  # Extraer ID (nombre del archivo sin extensión)
        
        # Cargar la imagen original una vez
        original_image = result.orig_img
        
        # Obtener las cajas delimitadoras en formato [x1, y1, x2, y2]
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        
        if boxes.size > 0:
            print(f"  > Imagen '{image_id}': {len(boxes)} estomas detectados.")
            detection_results[image_id] = (original_image, boxes)
        else:
            print(f"  > Imagen '{image_id}': No se detectaron estomas.")

    print("--- Etapa 1 Finalizada ---\n")
    return detection_results

def stage_2_crop_stomata(detection_results: dict, output_dir: Path):
    """
    Etapa 2: Recorta cada estoma detectado y lo guarda como una imagen individual.

    Args:
        detection_results (dict): Salida de la Etapa 1.
        output_dir (Path): Carpeta donde se guardarán los recortes.
    """
    print("--- Iniciando Etapa 2: Recorte de Estomas ---")
    total_crops = 0
    for image_id, (original_image, boxes) in detection_results.items():
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            
            # Recortar el estoma de la imagen original
            stoma_crop = original_image[y1:y2, x1:x2]
            
            # Crear un nombre de archivo único que conserve el ID original
            crop_filename = f"{image_id}_stoma_{i}.png"
            crop_path = output_dir / crop_filename
            
            # Guardar el recorte
            cv2.imwrite(str(crop_path), stoma_crop)
            total_crops += 1
    
    print(f"  > Se generaron {total_crops} recortes en '{output_dir}'.")
    print("--- Etapa 2 Finalizada ---\n")


def stage_3_segment_and_classify(cropped_dir: Path, model_path: Path, confidence: float):
    """
    Etapa 3: Aplica un modelo de segmentación a cada recorte para clasificar
             (abierto/cerrado) y obtener la máscara de segmentación.

    Args:
        cropped_dir (Path): Carpeta con las imágenes recortadas.
        model_path (Path): Ruta al modelo de segmentación .pt.
        confidence (float): Umbral de confianza para la segmentación.

    Returns:
        list: Una lista de diccionarios, cada uno con información sobre un estoma.
    """
    print("--- Iniciando Etapa 3: Segmentación y Clasificación ---")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo de segmentación no encontrado en: {model_path}")
    print("DEBUG YOLO MODEL PATH")
    print(model_path)
    model = YOLO(model_path)
    segmentation_results = []

    results_generator = model.predict(source=str(cropped_dir), conf=confidence, stream=True)

    for result in results_generator:
        if result.masks is None or len(result.masks) == 0:
            print(f"  > Advertencia: No se encontró segmentación para '{Path(result.path).name}'.")
            continue
            
        crop_path = Path(result.path)
        # Extraer ID original del nombre del recorte. Ej: "imagen_id_001_stoma_0" -> "imagen_id_001"
        original_id = "_".join(crop_path.stem.split("_")[:-2])

        # Asumimos una segmentación por recorte
        mask = result.masks[0]
        class_id = int(result.boxes.cls[0])
        class_name = model.names[class_id]
        
        # Obtener el polígono de la máscara como un array de numpy
        mask_polygon = mask.xy[0].astype(np.int32)
        
        segmentation_results.append({
            "original_id": original_id,
            "crop_path": crop_path,
            "class": class_name,
            "mask_polygon": mask_polygon
        })

    print(f"  > Se procesaron {len(segmentation_results)} recortes.")
    print("--- Etapa 3 Finalizada ---\n")
    return segmentation_results

def stage_4_extract_and_aggregate_parameters(segmentation_results: list, output_csv_path: Path):
    """
    Etapa 4: Calcula parámetros morfológicos (área) y los agrega por imagen original.

    Args:
        segmentation_results (list): Salida de la Etapa 3.
        output_csv_path (Path): Ruta para guardar el informe final en CSV.
    """
    print("--- Iniciando Etapa 4: Extracción y Agregación de Parámetros ---")
    
    # Calcular área para cada estoma
    for res in segmentation_results:
        res["area"] = cv2.contourArea(res["mask_polygon"])

    # Agrupar resultados por el ID de la imagen original
    aggregated_data = defaultdict(lambda: {
        "open_stomata_count": 0,
        "closed_stomata_count": 0,
        "total_area": 0.0,
        "stomata_details": []
    })

    for res in segmentation_results:
        image_id = res["original_id"]
        aggregated_data[image_id]["total_area"] += res["area"]
        aggregated_data[image_id]["stomata_details"].append(res)
        
        if "open" in res["class"]:
            aggregated_data[image_id]["open_stomata_count"] += 1
        elif "closed" in res["class"]:
            aggregated_data[image_id]["closed_stomata_count"] += 1

    # Preparar los datos para el DataFrame de Pandas
    final_report_data = []
    for image_id, data in aggregated_data.items():
        total_stomata = len(data["stomata_details"])
        average_area = (data["total_area"] / total_stomata) if total_stomata > 0 else 0
        
        final_report_data.append({
            "image_id": image_id,
            "total_stomata": total_stomata,
            "open_stomata_count": data["open_stomata_count"],
            "closed_stomata_count": data["closed_stomata_count"],
            "total_area_pixels": round(data["total_area"], 2),
            "average_area_pixels": round(average_area, 2)
        })

    # Crear y guardar el DataFrame
    if not final_report_data:
        print("  > No se generaron datos para el informe final.")
        return
        
    df = pd.DataFrame(final_report_data)
    df.to_csv(output_csv_path, index=False)
    
    print(f"  > Informe final guardado en '{output_csv_path}'.")
    print(df.to_string()) # Imprimir tabla en consola
    print("\n--- Etapa 4 Finalizada ---")