# Spanish Version

## Detección de Fake News

Proyecto de Machine Learning y Deep Learning para la detección automatizada de noticias falsas a partir de su contenido textual.  
Se han evaluado modelos clásicos (Logistic Regression, SVM, Random Forest, etc.) y avanzados (XGBoost, LightGBM, MLP).
Para ello se ha utilizado un dataset público de Kaggle.

### Objetivo

Desarrollar un sistema capaz de **clasificar noticias como reales o falsas**, analizando textos completos mediante técnicas de procesamiento de lenguaje natural (NLP) y algoritmos de clasificación supervisada.

### Estructura del proyecto

- `src/data_sample`: Contiene los datasets en bruto.
- `src/img`: Imágenes que se han utilizado.
- `src/notebooks`: Notebooks de análisis exploratorio y pruebas.
- `src/results_notebook`: Resultados, gráficos y visualizaciones.
- `src/models`: Modelos generados y automatizados durante la ejecución del código.
- `src/utils`: Código fuente organizado por tareas (módulos, funciones auxiliares, clases...).

```bash
ML_Fake_News/
│
├── data/                   # Dataset original
│
├── src/                    # Código fuente del proyecto
│   ├── data/               # Carga y preprocesamiento
│   ├── img/                # Gráficos generados automáticamente
│   ├── notebooks/          # Documentos de pruebas (si existen)
│   ├── result_notebook/    # Código fuente principal del proyecto (Resultados, comparativas y conclusiones)
│   ├── models/             # Entrenamiento y evaluación de modelos
│   └── utils/              # Funciones auxiliares (logs, visualizaciones, etc.)
│
├── requirements.txt        # Lista de dependencias
├── README.md               # Descripción general del proyecto
└── .gitignore              # Ficheros a ignorar por Git
```

### Modelos y Técnicas

    NLP: limpieza de texto, lematización, eliminación de stopwords
    Vectorización: TF-IDF
    Modelos evaluados:

        Regresión Logística
        SVM
        Random Forest
        XGBoost
        LightGBM
        MLPClassifier (scikit-learn)
        MLP (Keras + EarlyStopping)

    Evaluación:

        Accuracy
        Precision
        Recall
        F1-score
        Matriz de Confusión
        Análisis visual de resultados

### Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

### Autor

Proyecto desarrollado por Pablo como práctica avanzada de clasificación de texto y evaluación de modelos en ciencia de datos.


# English Version

## Fake News Detection

This project uses supervised machine learning and deep learning techniques to classify news articles as real or fake based on their textual content.

### Project Goal

To build an intelligent system that detects fake news using:

- Natural Language Processing (NLP)
- TF-IDF vectorization
- Multiple ML and DL classification models

### Project Structure

- `src/data_sample`: Contains the raw datasets.
- `src/img`: Images used or generated during the process.
- `src/notebooks`: Notebooks for exploratory data analysis and experiments.
- `src/results_notebook`: Results, charts, and visualizations.
- `src/models`: Models generated and saved during code execution.
- `src/utils`: Source code organized by task (modules, helper functions, classes...).

```bash
ML_Fake_News/
│
├── data/                   # Full dataset
├── src/                    # Source code
│   ├── data_sample/        # Dataset sample
│   ├── img/                # Auto-generated charts
│   ├── notebooks/          # EDA and training notebooks
│   ├── result_notebook/    # Results and visual comparisons
│   ├── models/             # Trained models
│   └── utils/              # Custom Python modules (EDA, plotting, evaluation)
│
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── .gitignore              # Git exclusions
```

### Models & Techniques

    NLP: text cleaning, lemmatization, stopword removal
    Vectorization: TF-IDF
    Models evaluated:

        Logistic Regression
        SVM
        Random Forest
        XGBoost
        LightGBM
        MLPClassifier
        MLP (Keras + EarlyStopping)

    Evaluation:

        Accuracy
        Precision
        Recall
        F1-score
        Confusion Matrix
        Visual analysis

### Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```
### Author

Project developed by Pablo as an advanced practice in text classification and model evaluation in data science.
