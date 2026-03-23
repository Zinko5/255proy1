# Proyecto de Análisis Predictivo: League of Legends

Este proyecto implementa los modelos de Machine Learning descritos en el documento `boceto.typ` utilizando la metodología CRISP-DM.

## Estructura del Código

- `modelado_lol.py`: Script principal que realiza todo el flujo de trabajo:
    - **Entendimiento de datos**: Resumen estadístico del dataset `league_data.csv`.
    - **Preparación de datos**: Limpieza de AFKs, remakes, normalización por tiempo y codificación One-Hot.
    - **Modelado**: Implementación de 6 modelos (Regresión Logística, Random Forest, SVM, KNN, XGBoost y Redes Neuronales).
    - **Evaluación**: Comparación de métricas (Accuracy, AUC) y generación de gráficos.
- `requirements.txt`: Lista de librerías necesarias.

## Cómo Ejecutar

1. Asegúrate de tener las dependencias instaladas:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecuta el script de modelado (análisis por consola):
   ```bash
   python modelado_lol.py
   ```
3. Ejecuta la plataforma interactiva (interfaz web):
   ```bash
   streamlit run app.py
   ```

## Objetivos Cumplidos

Siguiendo el `boceto.typ`:
- **Modelado Multialgoritmo**: 6 modelos implementados en `modelado_lol.py`.
- **Desarrollo Web**: Plataforma analítica con visualizaciones y simulador en `app.py`.
- **Preparación de Datos**: Normalización por tiempo y filtrado de AFKs.
- **Validación**: Métricas AUC y matrices de confusión automatizadas.
