
---

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


## 🚀 Cómo Usar en Google Colab (Recomendado)

Google Colab ofrece un entorno con GPU gratuita, lo que facilita enormemente la ejecución de este pipeline sin necesidad de configurar drivers o CUDA en tu computadora. Sigue estos pasos para ejecutar el proyecto en un notebook de Colab.

1.  **Abrir un Nuevo Notebook y Configurar la GPU**
    -   Ve a [Google Colab](https://colab.research.google.com/) y crea un nuevo notebook.
    -   En el menú, ve a `Entorno de ejecución` -> `Cambiar tipo de entorno de ejecución`.
    -   En "Acelerador por hardware", selecciona **GPU** y guarda.

2.  **Clonar el Repositorio y Descargar Modelos**
    Copia y pega el siguiente bloque de código en una celda del notebook y ejecútalo.
    ```python
    # Clonar el repositorio
    !git clone https://github.com/pythonmarti/Auto_Stomata.git
    %cd Auto_Stomata

    # Instalar Git LFS y descargar los modelos
    !git lfs install
    !git lfs pull
    ```

3.  **Instalar las Dependencias**
    Colab ya viene con una versión de PyTorch compatible con su GPU, por lo que solo necesitas instalar las dependencias del archivo `requirements_cuda.txt`.
    ```python
    # Instalar librerías necesarias
    !pip install -r requirements_cuda.txt

    # (Opcional) Verificar que la GPU está siendo detectada por PyTorch
    import torch
    print("GPU disponible:", torch.cuda.is_available())
    ```
    Si la salida es `True`, ¡estás listo para continuar!

4.  **Subir las Imágenes a Analizar**
    Tienes dos opciones para cargar tus imágenes:

    -   **Opción A (Manual):** En el panel izquierdo de Colab, ve a la pestaña `Archivos`. Navega dentro de la carpeta `Auto_Stomata/input_images`. Haz clic derecho en la carpeta `input_images` y selecciona `Subir`. Elige las imágenes desde tu computadora. *Nota: Estos archivos se borrarán cuando la sesión de Colab termine.*

    -   **Opción B (Conectando Google Drive - Recomendado):** Sube tus imágenes a una carpeta en tu Google Drive. Luego, monta tu Drive en Colab con el siguiente código y copia las imágenes al proyecto.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')

    # Reemplaza 'ruta/a/tus/imagenes' con la ruta real en tu Google Drive
    !cp /content/drive/MyDrive/ruta/a/tus/imagenes/* ./input_images/
    ```

5.  **Ejecutar el Pipeline**
    Ahora, ejecuta el script principal para procesar las imágenes.
    ```python
    !python main.py
    ```

6.  **Descargar los Resultados**
    Los resultados se guardarán en la carpeta `output`.

    -   **Para descargar a tu computadora:** En el panel de `Archivos`, navega a la carpeta `Auto_Stomata/output`. Haz clic derecho en los archivos que deseas descargar. Si son muchos, puedes comprimirlos primero:
    ```python
    # Comprimir la carpeta de resultados en un archivo .zip
    !zip -r output.zip output
    ```
    Luego, descarga el archivo `output.zip` desde el panel de `Archivos`.

    -   **Para guardar en Google Drive:**
    ```python
    # Reemplaza 'ruta/para/guardar/resultados' con la carpeta de destino en tu Drive
    !cp -r ./output/* /content/drive/MyDrive/ruta/para/guardar/resultados/
    ```

---


## 🏗️ Cómo Usar el Pipeline (local)

Sigue estos pasos para configurar y ejecutar el proyecto en tu máquina local.

#### 1. Clonar el repositorio
```bash
git clone https://github.com/pythonmarti/Auto_Stomata.git
cd Auto_Stomata
```

#### 2. Instalar Git LFS y descargar los modelos
Este proyecto utiliza Git LFS (Large File Storage) para manejar los archivos de los modelos.
```bash
git lfs install
git lfs pull
```

#### 3. Crear y activar un entorno virtual
Se recomienda encarecidamente crear un entorno virtual para aislar las dependencias del proyecto.

```bash
# Crear el entorno
python -m venv venv

# Activar en Windows
call venv/scripts/activate

# Activar en Mac/Linux
source venv/bin/activate
```

#### 4. Instalar dependencias (CPU o GPU)
Elige una de las dos opciones siguientes según tu hardware. La opción GPU es altamente recomendada para un procesamiento mucho más rápido.

<br>

##### Opción A: Configuración para CPU
Si no tienes una GPU NVIDIA o no deseas instalar CUDA, ejecuta el siguiente comando:
```bash
pip install -r requirements.txt
```
<br>

##### Opción B: Configuración para GPU (Recomendado, con NVIDIA CUDA)
Para utilizar la aceleración por GPU, necesitas tener una tarjeta gráfica NVIDIA compatible, los drivers actualizados, CUDA y la versión correcta de PyTorch.

<details>
<summary><b>🚀 Haz clic aquí para ver la guía detallada de instalación de CUDA y PyTorch para GPU (Windows)</b></summary>

---

##### **Paso 1: Verificar si ya tienes CUDA instalado**
1.  Abre la terminal (cmd o PowerShell).
2.  Escribe el comando:
    ```bash
    nvcc --version
    ```
    -   ✅ Si te muestra algo como `Cuda compilation tools, release 12.1`, ya tienes CUDA y puedes saltar al **Paso 4**.
    -   ❌ Si dice `"nvcc" no se reconoce...`, continúa con el siguiente paso.

##### **Paso 2: Determinar qué versión de CUDA puedes instalar**
1.  En la misma terminal, ejecuta:
    ```bash
    nvidia-smi
    ```
2.  En la esquina superior derecha, verás `CUDA Version: 12.2` (o un número similar). Esta es la **versión máxima** que tu driver soporta. Puedes instalar esa misma versión o una anterior (ej. 12.1, 11.8) que sea compatible con PyTorch.

##### **Paso 3: Instalar el NVIDIA CUDA Toolkit (si no lo tenías)**
1.  Ve al sitio oficial de descargas de CUDA: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
2.  Selecciona tu sistema operativo (Windows), arquitectura (x86\_64), versión y el tipo de instalador (`.exe [local]`).
3.  Descarga y ejecuta el instalador. Se recomienda elegir la instalación **Express (Recomendada)**.
4.  Una vez finalizado, reinicia tu PC y verifica la instalación de nuevo con `nvcc --version`.

##### **Paso 4: Instalar la versión de PyTorch compatible con tu CUDA**
1.  Visita la página oficial de PyTorch: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2.  Usa el configurador para seleccionar `Stable`, tu sistema operativo, `Pip`, `Python` y la versión de CUDA que instalaste (ej. CUDA 11.8 o 12.1).
3.  Copia el comando que te proporciona la página. Por ejemplo, para CUDA 11.8, sería:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    *Nota: Si el comando con `pip3` no funciona, prueba con `pip`.*

##### **Paso 5: Verificar que PyTorch detecta tu GPU**
1.  Abre una terminal de Python ejecutando `python`.
2.  Ingresa los siguientes comandos:
    ```python
    import torch
    print("CUDA disponible:", torch.cuda.is_available())
    print("Nombre de la GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU no detectada")
    ```
3.  ✅ Si `CUDA disponible` es `True` y ves el nombre de tu GPU, ¡la configuración ha sido un éxito! 🎉

---
</details>

<br>

Una vez que PyTorch con soporte CUDA esté instalado correctamente, instala el resto de las dependencias del proyecto:
```bash
pip install -r requirements_cuda.txt
```

#### 5. Colocar las imágenes a procesar
Dirígete a la carpeta `input_images` y coloca allí todas las imágenes microscópicas de estomas que desees analizar.

#### 6. Ejecutar la aplicación
Con el entorno virtual activado y desde la raíz del proyecto, ejecuta el script principal:
```bash
python main.py
```

#### 7. Revisar los resultados
Una vez que el script finalice, encontrarás todos los resultados en la carpeta `output`:
-   Las imágenes originales con las máscaras de segmentación superpuestas.
-   Un archivo Excel (`.xlsx`) con todos los parámetros morfológicos extraídos para cada estoma detectado en cada imagen.

---

*Nota: Los modelos fueron entrenados en un conjunto de datos de **Arabidopsis thaliana** bajo condiciones particulares de microscopía. No se garantiza un funcionamiento óptimo en imágenes con condiciones significativamente distintas a las del entrenamiento (otras especies, diferente iluminación, magnificación, etc.).*

## 📄 Licencia

Este proyecto se distribuye bajo la **Licencia MIT**.

Copyright (c) 2025, [Martina Lara Arriagada / Phytolearning Nucleo Milenio en Resilencia Vegetal]