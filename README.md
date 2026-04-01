# League Learning (DAT 255) - MLOps AI Analytics Pipeline

Este repositorio contiene la plataforma analítica predictiva de **League of Legends**, optimizada con metodologías de **MLOps** para gestión de datos (DVC) y experimentos (MLflow).

## 📁 Estructura del Proyecto

- `data/`: Datasets versionados (`league_data.csv`, `todosJugadores.csv`).
- `modelos/`: Artefactos de IA (escaladores, modelos por rol, importancia de variables).
- `metricas/`: Resultados históricos de rendimiento (JSONs y matrices de confusión).
- `documentos/`: Documentación oficial, informes en Typst e imágenes del proyecto.
- `codigo/`: Código fuente de la solución:
    - `app.py`: Plataforma interactiva en Streamlit con selector de versiones.
    - `modelado_lol.py`: Pipeline de entrenamiento automatizado con registro en MLflow.
    - `sacar_metricas.py`: Generador de reportes técnicos detallados.

## 🚀 Pipeline de Actualización (Flujo MLOps)

Para actualizar el sistema con nuevos datos y generar una nueva versión de la IA, siga estos pasos:

### 1. Actualizar Datos
Reemplace el archivo `data/league_data.csv` con el nuevo dataset. (Opcional: ejecute `dvc add data/league_data.csv` para versionar el cambio físico).

### 2. Generar nueva Versión de IA
Ejecute el pipeline de modelado. Esto creará automáticamente un registro en **MLflow** incluyendo un snapshot del dataset, los parámetros y los modelos resultantes.
```bash
cd codigo
source venv/bin/activate
python modelado_lol.py
```

### 3. Visualización
Inicie la aplicación Streamlit. La nueva versión aparecerá automáticamente en el selector de la barra lateral: **"🧠 Versión de Inteligencia"**.
```bash
cd codigo
streamlit run app.py
```

## 🧠 Metodología (CRISP-DM)

1.  **Business Understanding**: Jerarquizar factores de victoria (Feature Importance).
2.  **Data Understanding**: Análisis de 30+ variables competitivas.
3.  **Data Preparation**: Normalización per-minute, filtrado de AFKs y segmentación por rol.
4.  **Modeling**: Entrenamiento concurrente de 9 algoritmos (LR, RF, XGB, KNN, LDA, NB, DT, SVM, MLP).
5.  **Evaluation**: Validación cruzada y Permutation Importance para explicabilidad total.
6.  **Deployment**: Dashboard con simulador dinámico y perfil de invocador avanzado.

---
**Autor:** Gabriel Marcelo Muñoz Callisaya
**Materia:** DAT-255 | **Docente:** M. Sc. Menfy Morales Ríos

---
> **Legal Disclaimer:** League Learning isn't endorsed by Riot Games or associated properties. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.
