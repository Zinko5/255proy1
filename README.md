# League Learning (DAT 255) IA Analytics Pipeline

Este repositorio contiene el proyecto final de análisis predictivo basado en datos de League of Legends. El proyecto utiliza la metodología **CRISP-DM** para transformar 29,162 registros en inteligencia accionable.

## 📁 Estructura del Repositorio

- `boceto.typ`: Documentación formal del proyecto (Typst).
- `diccionario.md`: Guía de variables y métricas del dataset.
- `league_data.csv`: Dataset principal con estadísticas de partidas.
- `todosJugadores.csv`: Datos de ranking y perfil de invocadores.
- `codigo/`: Implementación técnica:
    - `app.py`: Plataforma interactiva comercial con Streamlit.
    - `modelado_lol.py`: Arquitectura de Machine Learning (6 modelos).
    - `requirements.txt`: Dependencias del proyecto.

## 🚀 Inicio Rápido

1.  Clona el repositorio:
    ```bash
    git clone git@github.com:Zinko5/league-learning.git
    cd league-learning
    ```
2.  Instala las dependencias:
    ```bash
    pip install -r codigo/requirements.txt
    ```
3.  Lanza la aplicación interactiva:
    ```bash
    cd codigo
    streamlit run app.py
    ```

## 🧠 Metodología (CRISP-DM)

1.  **Business Understanding**: Identificar factores de victoria en el meta actual.
2.  **Data Understanding**: Análisis exploratorio de 30+ variables desde la API de Riot Games.
3.  **Data Preparation**: Normalización por tiempo, filtrado de AFKs y One-Hot Encoding.
4.  **Modeling**: Comparativa entre Logistic Regression, Random Forest, SVM, KNN, XGBoost y MLP.
5.  **Evaluation**: Validación mediante AUC (Area Under Curve) y matrices de confusión.
6.  **Deployment**: Dashboard analítico para diagnóstico de partidas y visualización de perfiles.

---
**Autor:** Gabriel Marcelo Muñoz Callisaya
**Docente:** M. Sc. Menfy Morales Ríos
**Materia:** DAT-255

---
> **Legal Disclaimer:**
> **League Learning** isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.
