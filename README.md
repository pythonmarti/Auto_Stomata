# Auto-Stomata: Pipeline Automatizado para Análisis Morfológico de Estomas

Este repositorio contiene un pipeline de visión por computadora en Python, diseñado para la detección, segmentación y extracción automatizada de parámetros morfológicos de estomas a partir de imágenes de microscopía. La herramienta utiliza modelos de deep learning (YOLO) para procesar imágenes, y genera un reporte detallado con métricas clave.

### Visualización de Resultados del Modelo

| Ground Truth (Batch 0) | Predicciones (Batch 0) |
| :---: | :---: |
| ![val_batch0_labels](https://github.com/user-attachments/assets/b6d3cf89-af90-46cc-ab98-7cc575052edb) | ![val_batch0_pred](https://github.com/user-attachments/assets/630d8baf-f6f0-4954-b4d3-010e3350b2e3) |

## 1. ¿Qué son los estomas y por qué son importantes?

Los **estomas** son estructuras microscopicas ubicados principalmente en la epidermis de las hojas de las plantas. Cada estoma está rodeado por un par de células especializadas llamadas **células oclusivas** (o células de guarda), que controlan su apertura y cierre.

Su importancia es fundamental para la supervivencia y fisiología de la planta, ya que actúan como la principal interfaz entre la planta y la atmósfera, regulando procesos vitales:

-   **Intercambio Gaseoso:** Permiten la entrada de dióxido de carbono (CO₂) desde la atmósfera, que es el sustrato esencial para la **fotosíntesis**. Al mismo tiempo, liberan el oxígeno (O₂) producido durante este proceso.
-   **Transpiración:** Controlan la pérdida de agua en forma de vapor, un proceso conocido como transpiración. Esta regulación es crucial para mantener el equilibrio hídrico de la planta, especialmente bajo condiciones de estrés como la sequía.
-   **Termorregulación:** La transpiración ayuda a enfriar la superficie de la hoja, protegiendo a la planta del sobrecalentamiento por la radiación solar.

En resumen, los estomas son actores clave en el ciclo del carbono, el ciclo del agua y la respuesta de las plantas a su entorno. Su comportamiento influye directamente en la productividad agrícola, la resiliencia de los ecosistemas y los modelos climáticos globales.

## 2. ¿Cuál es la necesidad de extraer sus parámetros morfológicos?

La morfología de los estomas (su tamaño, forma, densidad y estado) no es estática. Varía entre especies, genotipos y en respuesta a factores ambientales como la luz, la humedad, la temperatura y la concentración de CO₂. La extracción y cuantificación de estos parámetros morfológicos nos permite:

-   **Entender Respuestas Fisiológicas:** El tamaño del poro estomático (apertura) se correlaciona directamente con la conductancia estomática, que determina la tasa de intercambio gaseoso y transpiración. Medir parámetros como el **área**, el **ancho** y el **largo** del poro permite inferir el estado fisiológico de la planta.
-   **Fenotipado de Alto Rendimiento (High-Throughput Phenotyping):** En programas de mejoramiento genético, es crucial analizar miles de plantas para seleccionar aquellas con rasgos deseables (ej. mayor eficiencia en el uso del agua). La **densidad estomática** (número de estomas por unidad de área) y el **índice de forma** son fenotipos clave que se correlacionan con la adaptación a diferentes climas.
-   **Estudios sobre el Cambio Climático:** Analizar cómo la morfología estomática responde a un aumento de CO₂ o a la sequía ayuda a predecir cómo se adaptarán los cultivos y los bosques a futuras condiciones ambientales.
-   **Diagnóstico de Estrés en Plantas:** Cambios anormales en el número de estomas cerrados o en su morfología pueden ser indicadores tempranos de estrés hídrico, nutricional o patogénico.

La cuantificación precisa de estos parámetros transforma observaciones cualitativas en datos cuantitativos robustos, esenciales para la investigación científica y la innovación agrícola.

## 3. Desafíos de la Extracción Manual de Parámetros Morfológicos

Tradicionalmente, la medición de parámetros estomáticos se ha realizado de forma manual o semi-manual utilizando software de análisis de imágenes como ImageJ. Este enfoque, aunque fundamental en el pasado, presenta importantes desafíos:

-   **Extremadamente Lento y Laborioso:** Un investigador puede pasar horas o incluso días delineando manualmente cientos de estomas en un conjunto de imágenes. Esto limita drásticamente la escala de los experimentos y el número de muestras que se pueden procesar.
-   **Subjetividad y Baja Reproducibilidad:** La definición de los bordes de un estoma puede variar significativamente entre diferentes investigadores, e incluso para el mismo investigador en diferentes momentos. Esta subjetividad introduce un sesgo que afecta la consistencia y reproducibilidad de los resultados.
-   **Propenso a Errores Humanos:** La fatiga y la falta de atención durante una tarea tan repetitiva pueden llevar a errores en la medición, el conteo o la clasificación (abierto/cerrado).
-   **Inviabilidad para el Fenotipado a Gran Escala:** Es prácticamente imposible aplicar métodos manuales en proyectos de fenotipado de alto rendimiento que requieren el análisis de miles o decenas de miles de imágenes.

**Auto-Stomata** aborda directamente estos desafíos al automatizar todo el flujo de trabajo. Al utilizar modelos de deep learning, este pipeline ofrece una solución **rápida, objetiva, reproducible y escalable**, permitiendo a los investigadores centrarse en la interpretación de los datos en lugar de en la tediosa tarea de la medición manual.

---

## 🏗️ Cómo Usar el Pipeline


1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/pythonmarti/Auto-Stomata.git
    cd Auto-Stomata
    ```
2.  **Instalar Git LFS**
    ```bash
    git lfs install
    git lfs pull
    ```
3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Dirigirse a carpeta input_images**
    Se dirige a carpeta "input_images" para subir las imagenes microscopicas de estomas
5.  **Ejecutar la aplicación:**
    ```bash
    python main.py
    ```
6.  **Resultados**
    Para visualizar las máscaras de segmentación realizadas por el modelo, junto al excel con la estimación de parametros morfológicos, revisar la carpeta "output"

**Modelos fueron entrenados para** *Arabidopsis thaliana* **, bajo ciertas condiciones particulares, no se garantiza funcionamiento optimo en imágenes con condiciones significativamente distintas a las del entrenamiento**

## 📄 Licencia

Este proyecto se distribuye bajo la **Licencia MIT**.

Copyright (c) 2025, [Martina Lara Arriagada /Phytolearning Nucleo Milenio en Resilencia Vegetal]
