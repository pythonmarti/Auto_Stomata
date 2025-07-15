# main.py
import config
from pipeline_stages import (
    stage_1_detect_stomata,
    stage_2_crop_stomata,
    stage_3_segment_and_classify,
    stage_4_extract_and_aggregate_parameters,
)

def run_pipeline():
    """
    Ejecuta el pipeline completo para el análisis de estomas.
    """
    print("=============================================")
    print("=== INICIANDO PIPELINE DE ANÁLISIS DE ESTOMAS ===")
    print("=============================================\n")
    
    # Etapa 1: Detección
    # La salida es un diccionario: {id_imagen: (imagen, [cajas])}
    detection_results = stage_1_detect_stomata(
        input_dir=config.INPUT_DIR,
        model_path=config.DETECTION_MODEL_PATH,
        confidence=config.DETECTION_CONFIDENCE
    )

    if not detection_results:
        print("El pipeline se detiene: No se detectaron estomas en ninguna imagen.")
        return

    # Etapa 2: Recorte
    # Esta etapa no devuelve datos, solo crea archivos en el disco.
    stage_2_crop_stomata(
        detection_results=detection_results,
        output_dir=config.CROPPED_OUTPUT_DIR
    )

    # Etapa 3: Segmentación y Clasificación
    # La salida es una lista de diccionarios, uno por cada estoma recortado.
    segmentation_results = stage_3_segment_and_classify(
        cropped_dir=config.CROPPED_OUTPUT_DIR,
        model_path=config.SEGMENTATION_MODEL_PATH,
        confidence=config.SEGMENTATION_CONFIDENCE
    )

    if not segmentation_results:
        print("El pipeline se detiene: No se pudo segmentar ningún estoma.")
        return

    # Etapa 4: Extracción de Parámetros y Reporte
    # Esta etapa genera el archivo CSV final.
    stage_4_extract_and_aggregate_parameters(
        segmentation_results=segmentation_results,
        output_csv_path=config.FINAL_REPORT_PATH
    )
    
    print("\n=============================================")
    print("===      PIPELINE FINALIZADO CON ÉXITO      ===")
    print("=============================================")

if __name__ == "__main__":
    run_pipeline()