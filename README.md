# 🧠 DCGAN para Generación de Imágenes de Gatos 🐱

Este proyecto implementa una Red Generativa Antagónica Profunda (DCGAN) utilizando PyTorch en Google Colab para generar imágenes de gatos a partir del dataset [`crawford/cat-dataset`](https://www.kaggle.com/datasets/crawford/cat-dataset) de Kaggle.

## 📌 Descripción General

El objetivo del proyecto es entrenar una red generativa adversaria profunda (DCGAN) para que aprenda a generar imágenes realistas de gatos. La arquitectura y el entrenamiento han sido optimizados para ejecutarse en Google Colab utilizando GPU.

## ⚙️ Tecnologías Utilizadas

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- PIL (Pillow)
- kagglehub
- Google Colab

## 🧾 Instrucciones para Ejecutar el Proyecto

1. **Sube tu archivo `kaggle.json` a Colab:**
   ```python
   from google.colab import files
   files.upload()  # Selecciona tu archivo kaggle.json
   ```

2. **Ejecuta todas las celdas del notebook `DCGAN_Gatos.ipynb`:**
   - Descarga el dataset automáticamente desde Kaggle usando `kagglehub`
   - Preprocesa las imágenes (redimensionado y normalización)
   - Construye la arquitectura DCGAN
   - Entrena el modelo (por defecto, 100 épocas)
   - Guarda imágenes generadas por época en la carpeta `output/`

3. **Visualiza resultados al final del entrenamiento**:
   - Se genera una galería con las imágenes falsas creadas en cada época
   - Permite observar la evolución del aprendizaje

## 🧠 Arquitectura del Modelo

### Generador
- Capas `ConvTranspose2D`
- Activaciones `ReLU`
- Normalización `BatchNorm2D`
- Activación final `Tanh`

### Discriminador
- Capas `Conv2D`
- Activaciones `LeakyReLU`
- Normalización `BatchNorm2D`
- Activación final `Sigmoid`


## 🛠️ Hiperparámetros

| Parámetro        | Valor     |
|------------------|-----------|
| Vector ruido     | 100       |
| Tamaño imagen    | 64x64     |
| Batch size       | 128       |
| Epochs           | 100       |
| Learning Rate    | 0.0002    |
| Optimizador      | Adam (β1 = 0.5) |


## 🔧 Optimizaciones Realizadas

- Uso de `kagglehub` para descargar directamente datasets en lugar de descomprimir manualmente.
- Uso de `ImageFolder` con transformaciones para normalizar las imágenes.
- Batch size ajustado para balance entre rendimiento y uso de memoria en GPU.
- Generación periódica de imágenes durante el entrenamiento para visualización inmediata.

## 🖼️ Resultados

También se incluye un script que permite visualizar las imágenes de las épocas 1 a 99 fácilmente.
