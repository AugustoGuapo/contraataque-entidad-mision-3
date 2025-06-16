# 🛰️ Proyecto Misión 3 - Contraataque a La Entidad
## Autores:
- Augusto Sergio Diaz Polo
- Maryi Rodriguez Brieva

**Clasificación de imágenes manipuladas con redes neuronales convolucionales en entornos de capacidad limitada**

## 🛡️ Contexto de la Misión

Misión 3 forma parte del programa de defensa computacional del Comando de Resistencia Digital de la Universidad de Córdoba, desarrollado bajo estrictos lineamientos de investigación aplicada en entornos simulados de guerra informativa.

Ante el avance de *La Entidad* —una fuerza digital encubierta dedicada a manipular evidencias visuales—, el Departamento de Ciberinteligencia fue asignado a la tarea de construir un sistema de detección autónoma de imágenes falsificadas. Este repositorio representa el despliegue técnico de dicho sistema.

## 🎯 Objetivos Operativos

- Construir una red neuronal convolucional capaz de distinguir entre imágenes **originales** y **manipuladas**.
- Mantener viabilidad de entrenamiento en entornos con **hardware limitado** (1 GPU de gama media y 16GB de RAM).
- Implementar y comparar distintas arquitecturas para maximizar **precisión, recall y F1-score**.
- Aplicar estrategias avanzadas de preprocesamiento, balanceo, *data augmentation* y optimización de hiperparámetros.

## 🧠 Contenido del Repositorio

```
├── src/
│   ├── main/
│   │   ├── java/                      # Implementación completa de los modelos DL4J
│   │   └── resources/
│   │       ├── dataset/              # Dataset sin divisiones 
│   │       ├── dataset_final/        # Dataset dividido en train/test
│   │       ├── saved_models/         # Modelos entrenados (.zip)
│   │       └── muestreador.py        # Script básico para dividir en train/test el dataset

└── README.md
```
## 🧬 Arquitecturas Desplegadas

Se entrenaron 5 arquitecturas distintas, variando:

- Cantidad y tipo de capas (convolucionales, densas, normalización por lotes, *dropout*)
- Funciones de activación (`ReLU`, `Sigmoid`, `LeakyReLU`, `Softmax`)
- Estrategias de regularización (`L2`, *dropout*)
- Ajuste de tasa de aprendizaje dinámico (`MapSchedule`)
- Pesos personalizados por clase para mitigar desbalance

## 📦 Dataset Clasificado

- **Origen:** Imágenes reales y manipuladas curadas por el agente de campo
- **Preprocesamiento:**
    - Reescalado a `80x70` píxeles
    - Escala de grises (1 canal)
    - Formato uniforme `.jpg`
- **Augmentación:** Brillo, rotación, desenfoque gaussiano

## ⚙️ Requisitos Técnicos

- Java 11 (probado con Amazon Corretto)
- Maven 3.8+
- DL4J 1.0.0-M1 (con backend CUDA 11.2)
- CUDNN 8.1
- NVIDIA RTX 3060 o superior (11.2 compute capability requerido)
- Python 3.10+ (para scripts auxiliares)

> ⚠️ El modelo no ha sido optimizado para ejecutarse en entornos sin GPU. Se recomienda entorno con al menos 6GB de VRAM.

## 🔥 Entrenamiento con CUDA

Este repositorio incluye un `pom.xml` que activa la aceleración por GPU mediante:

```xml
<artifactId>deeplearning4j-cuda-11.2</artifactId>
<version>1.0.0-M1</version>
```

## 🧾 Licencia y Código de Ética
Este sistema fue desarrollado únicamente con fines educativos y de defensa digital. Cualquier uso fuera de este marco no está autorizado ni respaldado por el equipo responsable.