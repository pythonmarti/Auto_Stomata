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

def stage_1_detect_stomata(image_path: Path, model_path: Path, confidence: float):
    """
    Etapa 1: Detecta estomas en UNA SOLA imagen de entrada usando un modelo YOLO.

    Args:
        image_path (Path): Ruta a la imagen de entrada individual.
        model_path (Path): Ruta al modelo de detección .pt.
        confidence (float): Umbral de confianza para las detecciones.

    Returns:
        dict: Un diccionario con el ID de la imagen y una tupla 
              (imagen_original, cajas_delimitadoras), o un diccionario vacío si no hay detecciones.
    """
    print(f"--- Etapa 1: Detección en '{image_path.name}' ---")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo de detección no encontrado en: {model_path}")

    model = YOLO(model_path)
    detection_results = {}

    # Predecir en la imagen individual
    result = model.predict(source=str(image_path), conf=confidence, verbose=False)[0] # verbose=False para un log más limpio

    image_id = image_path.stem
    original_image = result.orig_img
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    
    if boxes.size > 0:
        print(f"  > Se detectaron {len(boxes)} estomas.")
        detection_results[image_id] = (original_image, boxes)
    else:
        print(f"  > No se detectaron estomas con confianza >= {confidence}.")

    return detection_results

def stage_2_crop_stomata(detection_results: dict, output_dir: Path):
    """
    Etapa 2: Recorta cada estoma detectado y lo guarda como una imagen individual.
    Esta función no necesita cambios, ya funciona para los datos de una imagen a la vez.
    """
    print(f"--- Etapa 2: Recortando estomas para '{list(detection_results.keys())[0]}' ---")
    total_crops = 0
    for image_id, (original_image, boxes) in detection_results.items():
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            stoma_crop = original_image[y1:y2, x1:x2]
            
            if stoma_crop.size == 0:
                continue

            crop_filename = f"{image_id}_stoma_{i}.png"
            crop_path = output_dir / crop_filename
            cv2.imwrite(str(crop_path), stoma_crop)
            total_crops += 1
    
    print(f"  > Se generaron {total_crops} recortes en '{output_dir}'.")

def stage_3_segment_and_classify(cropped_dir: Path, model_path: Path, confidence: float, masks_output_dir: Path):
    """
    Etapa 3: Aplica segmentación, guarda visualización y extrae datos.
    Esta función no necesita cambios, ya funciona con directorios de entrada/salida específicos.
    """
    print(f"--- Etapa 3: Segmentación y Clasificación ---")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo de segmentación no encontrado en: {model_path}")

    model = YOLO(model_path)
    segmentation_results = []
    
    # Lista de archivos en el directorio de recortes
    crop_files = list(cropped_dir.glob("*.[jp][pn]g"))
    if not crop_files:
        print("  > No hay recortes para procesar en esta etapa.")
        return []

    # Usamos verbose=False para evitar el log detallado de cada imagen de recorte
    results_generator = model.predict(source=str(cropped_dir), conf=confidence, stream=True, verbose=False)

    for result in results_generator:
        if result.masks is None or len(result.masks) == 0:
            continue
            
        crop_path = Path(result.path)
        original_id = "_".join(crop_path.stem.split("_")[:-2])
        mask = result.masks[0]
        class_id = int(result.boxes.cls[0])
        class_name = model.names[class_id]
        mask_polygon = mask.xy[0].astype(np.int32)
        
        crop_image = cv2.imread(str(crop_path))
        color = (0, 255, 0) if "open" in class_name else (0, 0, 255)
        cv2.drawContours(crop_image, [mask_polygon], -1, color, 2)
        
        confidence_score = float(result.boxes.conf[0])
        label = f"{class_name} ({confidence_score:.2f})"
        cv2.putText(crop_image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        output_mask_path = masks_output_dir / crop_path.name
        cv2.imwrite(str(output_mask_path), crop_image)

        segmentation_results.append({
            "original_id": original_id,
            "crop_path": crop_path,
            "class": class_name,
            "mask_polygon": mask_polygon
        })

    print(f"  > Se procesaron y segmentaron {len(segmentation_results)} recortes.")
    print(f"  > Visualizaciones de máscaras guardadas en '{masks_output_dir}'.")
    return segmentation_results

def stage_4_extract_and_aggregate_parameters(segmentation_results: list, output_csv_path: Path):
    """
    Etapa 4: Calcula parámetros morfológicos y los agrega por imagen original.
    Esta función no necesita cambios, ya está diseñada para agregar datos de múltiples imágenes.
    """
    print("--- Etapa 4: Extracción y Agregación de Parámetros Globales ---")
    
    if not segmentation_results:
        print("  > No hay datos de segmentación para generar el informe.")
        return

    for res in segmentation_results:
        res["area"] = cv2.contourArea(res["mask_polygon"])

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

    final_report_data = []
    for image_id, data in sorted(aggregated_data.items()): # sorted() para un orden consistente
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
        
    df = pd.DataFrame(final_report_data)
    df.to_csv(output_csv_path, index=False)
    
    print(f"  > Informe final con {len(df)} filas guardado en '{output_csv_path}'.")
    print("--- Resumen del Informe ---")
    print(df.to_string())