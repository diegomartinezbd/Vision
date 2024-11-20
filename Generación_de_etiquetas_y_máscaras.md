# Estimado estudiante:

En el proyecto final debe entrenar una red convolucional para detección de objetos y otra para segmentación, utilizando una plataforma de bajo consumo de recursos, como TensorFlow Lite. Sin embargo, antes de comenzar, es indispensable contar con los respectivos conjuntos de imágenes.

## Actividades para la sesión de hoy

1. **Selección del dataset:**
   - Escoja un conjunto de imágenes, por ejemplo, de Kaggle, con una resolución mínima de 128x128 píxeles. Estas imágenes deben contener el **objeto** que desea detectar o segmentar.
   - Tenga en cuenta que un mayor tamaño de las imágenes incrementará el tiempo de entrenamiento. Por lo tanto, si es necesario, puede reducir su resolución.

2. **Generación de etiquetas para detección:**
   - Utilice **Grounding DINO**, que permite localizar un objeto a partir de una entrada textual, para generar las etiquetas en formato YOLO que identifiquen la ubicación del **objeto** seleccionado.

3. **Obtención de imágenes de segmentación:**
   - Emplee el modelo **Segment Anything Model (SAM)** para generar máscaras de segmentación que distingan claramente entre el **objeto** y el fondo.

## Instrucciones para el trabajo en grupo

- El trabajo debe realizarse en grupos de **máximo dos estudiantes**.
- Cada grupo debe crear una carpeta en el repositorio de la clase y subir el trabajo realizado durante la sesión de hoy dentro del horario asignado.
- Podrán seguir mejorando su proyecto hasta la próxima clase.

## Resultado Esperado

- El resultdo final debe ser una archivo .ipynb que se pueda ejecutar en Colab.
- Luego de ejecutar el código deben haber tres subcarpetas en Colab.
  - **images**: Imagenes del dataset, todas del mismo tamaño y sin perder el aspecto de la imagen.
  - **detection**: Etiquetas de cada una de las imágenes en formato YOLO.
  - **segmentation**: Imágenes con las respectiva máscaras.

## Referencias recomendadas

- Páginas oficiales y tutoriaes de **Grounding DINO** y **Segment Anything Model (SAM)**.
- https://github.com/GerardoMunoz/Vision/blob/main/Blanqueamiento_Coralino_Carol_Fernandez.pdf
