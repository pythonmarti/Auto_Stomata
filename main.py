# main.py
import config
from pipeline_stages import (
    stage_1_detect_stomata,
    stage_2_crop_stomata,
    stage_3_segment_and_classify,
    stage_4_extract_and_aggregate_parameters,
)
from pathlib import Path
import shutil

def run_pipeline():
    """
    Ejecuta el pipeline completo, iterando sobre cada imagen en el directorio de entrada
    y organizando las salidas en subcarpetas por imagen.
    """
    print("=============================================")
    print("=== INICIANDO PIPELINE DE ANÁLISIS DE ESTOMAS ===")
    print("=============================================\n")

    # Limpiar la carpeta de salida principal para evitar resultados antiguos
    for path in config.OUTPUT_DIR.iterdir():
        if path.is_dir():
            print(f"Limpiando directorio antiguo: {path}")
            shutil.rmtree(path)
        # También limpiar el reporte antiguo para no confundir
        elif path.name == config.FINAL_REPORT_PATH.name:
            print(f"Limpiando reporte antiguo: {path}")
            path.unlink()

    # Buscar todas las imágenes en el directorio de entrada
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    input_images = []
    for ext in image_extensions:
        input_images.extend(config.INPUT_DIR.glob(ext))

    if not input_images:
        print(f"ERROR: No se encontraron imágenes en el directorio '{config.INPUT_DIR}'.")
        print("Asegúrate de colocar tus archivos .jpg, .jpeg o .png allí.")
        return

    print(f"Se encontraron {len(input_images)} imágenes para procesar.\n")
    
    # Listas y diccionarios para acumular los resultados de TODAS las imágenes
    all_images_segmentation_results = []
    all_images_detection_results = {} # Necesario para la Etapa 4

    # Bucle principal: procesar cada imagen de forma independiente
    for image_path in input_images:
        image_id = image_path.stem
        print(f"=============================================")
        print(f"=== PROCESANDO IMAGEN: {image_id} ===")
        print(f"=============================================")

        # --- Crear directorios de salida específicos para esta imagen ---
        image_output_dir = config.OUTPUT_DIR / image_id
        cropped_output_dir = image_output_dir / "2_cropped_stomata"
        masks_output_dir = image_output_dir / "3_segmentation_masks"
        
        cropped_output_dir.mkdir(parents=True, exist_ok=True)
        masks_output_dir.mkdir(parents=True, exist_ok=True)

        # --- Ejecutar Etapas 1, 2 y 3 para la imagen actual ---
        
        # Etapa 1: Detección
        detection_results = stage_1_detect_stomata(
            image_path=image_path,
            model_path=config.DETECTION_MODEL_PATH,
            confidence=config.DETECTION_CONFIDENCE
        )

        if not detection_results:
            print(f"No se detectaron estomas en '{image_id}', pasando a la siguiente imagen.\n")
            continue
        
        # Acumular los resultados de la detección para usarlos en la Etapa 4
        all_images_detection_results.update(detection_results)

        # Etapa 2: Recorte
        stage_2_crop_stomata(
            detection_results=detection_results,
            output_dir=cropped_output_dir
        )

        # Etapa 3: Segmentación y Clasificación
        # Se pasa el nuevo parámetro 'save_visualizations' desde el archivo de configuración
        segmentation_results_for_image = stage_3_segment_and_classify(
            cropped_dir=cropped_output_dir,
            model_path=config.SEGMENTATION_MODEL_PATH,
            confidence=config.SEGMENTATION_CONFIDENCE,
            masks_output_dir=masks_output_dir,
            save_visualizations=config.SAVE_MASK_VISUALIZATIONS # <-- CORRECCIÓN: El parámetro va aquí
        )

        # Acumular los resultados de segmentación de esta imagen en la lista global
        if segmentation_results_for_image:
            all_images_segmentation_results.extend(segmentation_results_for_image)
        
        print(f"--- Fin del procesamiento para la imagen: {image_id} ---\n")

    # --- Etapa 4: Ejecutar una sola vez al final con todos los resultados acumulados ---
    print("=============================================")
    print("===   GENERANDO REPORTE FINAL AGREGADO    ===")
    print("=============================================")
    
    # Se pasan ambos, los resultados de segmentación y los de detección
    stage_4_extract_and_aggregate_parameters(
        segmentation_results=all_images_segmentation_results,
        detection_results=all_images_detection_results, # <-- Se pasa el diccionario de detecciones
        output_csv_path=config.FINAL_REPORT_PATH
    )
    
    print("\n=============================================")
    print("===      PIPELINE FINALIZADO CON ÉXITO      ===")
    print("=============================================")

if __name__ == "__main__":
    run_pipeline()