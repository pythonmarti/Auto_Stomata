# pipeline_stages.py
import cv2
import torch
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import distance
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

    result = model.predict(source=str(image_path), conf=confidence, verbose=False)[0]

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
    No se necesitan cambios en esta función.
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


def stage_3_segment_and_classify(cropped_dir: Path, model_path: Path, confidence: float, masks_output_dir: Path, save_visualizations: bool):
    """
    Etapa 3: Aplica segmentación, calcula parámetros morfológicos individuales y, opcionalmente, guarda visualizaciones.

    NUEVO: El argumento 'save_visualizations' controla si se crean las imágenes de máscaras.
    """
    print(f"--- Etapa 3: Segmentación y Cálculo de Parámetros Individuales ---")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo de segmentación no encontrado en: {model_path}")

    model = YOLO(model_path)
    segmentation_results = []
    
    crop_files = list(cropped_dir.glob("*.[jp][pn]g"))
    if not crop_files:
        print("  > No hay recortes para procesar en esta etapa.")
        return []
    
    results_generator_extra_params = {}

    if config.IS_CUDA_AVAILABLE :
        print("[INFO] CUDA DETECTED, adding to extra params")
        results_generator_extra_params["device"] = "cuda"
        #{"device": "cuda"}

    elif config.IS_MPS_AVAILABLE:
        print("[INFO] MPS DETECTED, adding to extra params")
        results_generator_extra_params["device"] = "mps"

    results_generator = model.predict(
        source=str(cropped_dir),
        conf=confidence,
        stream=True,
        verbose=False,
        **results_generator_extra_params
    )
    for result in results_generator:
        if result.masks is None or len(result.masks) == 0:
            continue
            
        crop_path = Path(result.path)
        original_id = "_".join(crop_path.stem.split("_")[:-2])
        mask = result.masks[0]
        class_id = int(result.boxes.cls[0])
        class_name = model.names[class_id]
        mask_polygon = mask.xy[0].astype(np.int32)
        
        
        area = cv2.contourArea(mask_polygon)
        if len(mask_polygon) >= 5:
            (x, y), (w, h), angle = cv2.minAreaRect(mask_polygon)
            length = max(w, h)
            width = min(w, h)
            shape_index = width / length if length > 0 else 0
        else:
            length, width, shape_index = 0, 0, 0

        if save_visualizations:
            # Este bloque solo se ejecuta si el parámetro es True
            crop_image = cv2.imread(str(crop_path))
            overlay = crop_image.copy()
            color = (0, 255, 0) if "open" in class_name else (0, 0, 255)
            cv2.fillPoly(overlay, [mask_polygon], color)
            alpha = 0.4
            image_with_mask = cv2.addWeighted(overlay, alpha, crop_image, 1 - alpha, 0)
            cv2.drawContours(image_with_mask, [mask_polygon], -1, color, 1)
            confidence_score = float(result.boxes.conf[0])
            class_abbreviation = "O" if "open" in class_name else "C"
            label = f"{class_abbreviation} ({confidence_score:.2f})"
            cv2.putText(image_with_mask, label, (2, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
            output_mask_path = masks_output_dir / crop_path.name
            cv2.imwrite(str(output_mask_path), image_with_mask)

        segmentation_results.append({
            "original_id": original_id,
            "crop_path": crop_path,
            "class": class_name,
            "area": area,
            "length": length,
            "width": width,
            "shape_index": shape_index
        })

    print(f"  > Se procesaron y segmentaron {len(segmentation_results)} recortes.")
    
    if save_visualizations:
        print(f"  > Visualizaciones de máscaras guardadas en '{masks_output_dir}'.")
    else:
        print("  > Se omitió la creación de archivos de visualización de máscaras (configuración).")
        
    return segmentation_results

def _calculate_distribution_pattern(boxes):
    """
    Función auxiliar para calcular el patrón de distribución de estomas.
    Utiliza el Coeficiente de Variación (CV) de la distancia al vecino más cercano.
    CV < 1 -> Uniforme, CV ≈ 1 -> Aleatorio, CV > 1 -> Agrupado.

    Args:
        boxes (np.array): Array de cajas delimitadoras (N, 4) con formato (x1, y1, x2, y2).

    Returns:
        float: El valor del Coeficiente de Variación. Retorna np.nan si no es calculable.
    """
    if len(boxes) < 2:
        return np.nan 

    
    centroids = np.array([((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes])
    dist_matrix = distance.cdist(centroids, centroids, 'euclidean')
    np.fill_diagonal(dist_matrix, np.inf)
    
    nearest_neighbor_distances = np.min(dist_matrix, axis=1)
    
    mean_dist = np.mean(nearest_neighbor_distances)
    std_dev_dist = np.std(nearest_neighbor_distances)
    

    cv = std_dev_dist / mean_dist if mean_dist > 0 else 0.0
    
    return cv

def stage_4_extract_and_aggregate_parameters(segmentation_results: list, detection_results: dict, output_csv_path: Path):
    """
    Etapa 4: Agrega los parámetros morfológicos por imagen original, calcula el patrón de distribución
             y genera el informe CSV final.
    
    NUEVO: Agrega longitud, ancho, índice de forma y calcula el patrón de distribución.
    """
    print("--- Etapa 4: Extracción y Agregación de Parámetros Globales ---")
    
    if not segmentation_results:
        print("  > No hay datos de segmentación para generar el informe.")
        return

    aggregated_data = defaultdict(lambda: {
        "open_stomata_count": 0,
        "closed_stomata_count": 0,
        "total_area": 0.0,
        "total_length": 0.0,
        "total_width": 0.0,
        "total_shape_index": 0.0,
        "stomata_count": 0
    })

    for res in segmentation_results:
        image_id = res["original_id"]
        data = aggregated_data[image_id]

        data["total_area"] += res["area"]
        data["total_length"] += res["length"]
        data["total_width"] += res["width"]
        data["total_shape_index"] += res["shape_index"]
        data["stomata_count"] += 1
        
        if "open" in res["class"]:
            data["open_stomata_count"] += 1
        elif "closed" in res["class"]:
            data["closed_stomata_count"] += 1

    final_report_data = []
    for image_id, data in sorted(aggregated_data.items()):
        count = data["stomata_count"]
        avg_area = data["total_area"] / count if count > 0 else 0
        avg_length = data["total_length"] / count if count > 0 else 0
        avg_width = data["total_width"] / count if count > 0 else 0
        avg_shape_index = data["total_shape_index"] / count if count > 0 else 0
        
        boxes_for_image = detection_results.get(image_id, (None, []))[1]
        distribution_cv = _calculate_distribution_pattern(boxes_for_image)

        final_report_data.append({
            "image_id": image_id,
            # NOTA: Esto es un conteo por imagen. La densidad real (estomas/mm^2) requeriría una escala.
            "stomata_density_count": count, 
            "open_stomata_count": data["open_stomata_count"],
            "closed_stomata_count": data["closed_stomata_count"],
            "average_area_pixels": round(avg_area, 2),
            "average_length_pixels": round(avg_length, 2),
            "average_width_pixels": round(avg_width, 2),
            "average_shape_index": round(avg_shape_index, 3), # Más decimales para el índice
            "distribution_pattern_cv": round(distribution_cv, 3) if not np.isnan(distribution_cv) else 'N/A'
        })
        
    df = pd.DataFrame(final_report_data)
    df.to_csv(output_csv_path, index=False)
    
    print(f"  > Informe final con {len(df)} filas guardado en '{output_csv_path}'.")
    print("--- Resumen del Informe ---")
    print(df.to_string())