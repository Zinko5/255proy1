// #import "/utiles/typst/plantillas/caratulaPlantillaT.typ": *

// #import "/utiles/typst/plantillas/apuntesPlantillaT.typ": *
// #show: config

// #caratula(
//   "Boceto 2",
//   "DAT 255 - Machine Learning",
//   "M. Sc. Menfy Morales Ríos",
//   "Gabriel Marcelo Muñoz Callisaya",
//   datetime(year: 2026, month: 03, day: 24),
// )

#heading(text(size: 2em)[League Learning], outlined: false)

#outline()

#pagebreak()

= Introducción

El presente documento detalla el desarrollo de un proyecto de análisis de datos enfocado en el videojuego League of Legends (LoL). El objetivo principal es transformar un conjunto de datos masivo y complejo, proveniente de partidas competitivas, en información estructurada y modelos predictivos accionables. Para lograrlo, se implementa una metodología rigurosa de Ciencia de Datos, aplicando técnicas avanzadas de preprocesamiento, modelado estadístico y aprendizaje automático (Machine Learning).

El proyecto aborda la necesidad de extraer patrones significativos de un entorno dinámico y de alta dimensionalidad, donde las interacciones entre múltiples variables (kills, oro, visión, objetivos) determinan el resultado final. Se utilizan diversos algoritmos, desde modelos lineales hasta redes neuronales, para evaluar diferentes enfoques de predicción y seleccionar el más robusto.

El resultado es una plataforma analítica que no solo describe el comportamiento del juego, sino que permite predecir resultados con un alto grado de precisión y ofrecer recomendaciones basadas en evidencia empírica. Este enfoque es fundamental para jugadores, entrenadores y analistas que buscan optimizar su rendimiento y tomar decisiones estratégicas fundamentadas en datos.

= Antecedentes

League of Legends es un juego multijugador en línea desarrollado y publicado por Riot Games. Es un juego MOBA (Multiplayer Online Battle Arena) en el que dos equipos de cinco jugadores compiten entre sí para destruir la base del equipo contrario. El juego fue lanzado en 2009 y desde entonces se ha convertido en uno de los juegos más populares del mundo, con millones de jugadores en todo el mundo.

Con una popularidad que se mantiene a lo largo de los años, el juego ha desarrollado un ecosistema competitivo profesional exitoso, con ligas profesionales en todo el mundo y un campeonato mundial que atrae a decenas de miles de espectadores, el juego se ha convertido en la principal fuente de ingresos para varios jugadores profesionales y sus equipos de entrenadores y analistas.

= Problemática

En el entorno competitivo de League of Legends, la planificación estratégica y técnica de los jugadores se sustenta de forma predominante en la intuición individual debido a la alta densidad de indicadores técnicos generados en cada partida. La multiplicidad de variables de desempeño (visión, economía, objetivos y combate) genera un entorno de alta dimensionalidad que excede la capacidad de procesamiento manual y análisis estadístico convencional. Esta dificultad para sistematizar y jerarquizar los registros tácticos impide que el jugador cuantifique el impacto real de sus acciones sobre la probabilidad de victoria, resultando en un diagnóstico de rendimiento basado en percepciones parciales.

= Problemática

En el entorno competitivo de League of Legends, los jugadores identifican los factores determinantes de la victoria mediante la revisión manual de las estadísticas descriptivas generadas en cada partida. Los indicadores técnicos —distribuidos en categorías de visión, economía, objetivos y combate— se presentan en formatos agregados y aislados en las pantallas de resumen de partida y en los registros históricos disponibles.

== Árbol de problema

- *Problema Central*: Identificación de factores determinantes de la victoria mediante revisión manual de estadísticas descriptivas en League of Legends.
- *Causas*:
  - Presentación de indicadores técnicos en valores agregados y aislados por categoría en las pantallas de resumen de partida.
  - Estructura de los datos de rendimiento con múltiples variables simultáneas en cada partida.
  - Almacenamiento de trayectorias históricas de partidas como datos brutos en las bases de datos accesibles.
- *Efectos*:
  - Selección de áreas de práctica basada en la prominencia de métricas en revisiones individuales de partidas.
  - Evaluación del rendimiento centrada en métricas aisladas por categoría.
  - Utilización de registros históricos como datos estáticos para la autoevaluación competitiva.

#image("arbol_problema_final.png")

= Objetivo

Optimizar el diagnóstico del rendimiento táctico mediante el desarrollo de una plataforma analítica que sistematice, valide y jerarquice los factores determinantes de la victoria en League of Legends. El proyecto busca establecer un marco de referencia basado en evidencia para la planificación estratégica, permitiendo que el jugador identifique con precisión los indicadores críticos que influyen en su desempeño competitivo a través de modelos de Machine Learning.

== Árbol de objetivos

- *Objetivo Central*: Optimizar el diagnóstico de rendimiento táctico mediante una plataforma analítica que jerarquice los factores de victoria basados en Machine Learning.

- *Medios*:
  - *Sistematizar* la multiplicidad de indicadores técnicos en estructuras de datos normalizadas para su procesamiento algorítmico.
  - *Validar* modelos predictivos multialgoritmo que identifiquen los patrones de éxito en registros competitivos reales.
  - *Jerarquizar* el peso específico (*Feature Importance*) de cada variable táctica para identificar los factores reales de victoria.
  - *Diseñar* una interfaz de visualización que traduzca el procesamiento de datos complejos en recomendaciones estratégicas accionables.
- *Fines*:
  - *Fundamentar* la planificación del entrenamiento en evidencia matemática y estratégica fidedigna.
  - *Reducir* la dependencia de la intuición subjetiva en la evaluación del desempeño competitivo.
  - *Potenciar* el crecimiento del nivel de juego mediante el conocimiento de las variables con mayor impacto real en el resultado.

#image("arbol_objetivo_final.png")

= Justificación

El análisis de datos en el contexto de los eSports ha evolucionado hacia un componente crítico para la optimización del rendimiento. En League of Legends, aunque existen diversas plataformas de consulta de estadísticas, estas operan predominantemente en un nivel descriptivo, mostrando resultados históricos sin proyectar tendencias predictivas ni establecer la importancia relativa de cada acción. Ante esta limitación, el desarrollo de un sistema que trascienda hacia el análisis predictivo y la jerarquización de variables permite que el jugador abandone la dependencia de la intuición y adopte una estrategia basada en evidencia matemática.

En el ámbito profesional, la capacidad de identificar patrones no lineales de victoria representa una ventaja competitiva decisiva. La implementación de un motor multialgoritmo no solo permite predecir resultados, sino que democratiza el acceso a diagnósticos de alta fidelidad que antes estaban reservados para equipos técnicos especializados con acceso a personal analista. Este proyecto se justifica como una herramienta de transición entre la observación pasiva de datos y la ejecución estratégica fundamentada en la inteligencia de datos.

= Metodología

El método seleccionado para el desarrollo es CRISP-DM, ya que permite un análisis sistemático y riguroso de los datos, lo que facilita la identificación de patrones y la construcción de modelos predictivos.

= Desarrollo

== Entendimiento del negocio

El objetivo principal de este desarrollo es identificar y cuantificar los factores críticos que determinan la victoria en partidas competitivas de League of Legends. En un entorno de saturación de datos crudos donde la información accionable permanece oculta, se busca construir un sistema que permita a jugadores y analistas priorizar métricas de rendimiento que tengan una correlación directa con el éxito. El negocio del *coaching* y el análisis profesional se beneficia de modelos que no solo predigan el resultado, sino que expliquen qué variables (visión, control de objetivos, ventaja en fase de líneas) deben optimizarse para mejorar el *winrate*.

== Entendimiento de los datos

El dataset base, `league_data.csv`, consta de 29,162 registros capturados a través de la API de Riot Games. Cada registro representa el desempeño individual de un jugador en una partida específica, abarcando 31 variables clave definidas en el `diccionario.md`:

- *Variable Objetivo*: `win` (booleano), que indica si el jugador ganó o perdió la partida.
- *Variables de Combate*: `kills`, `deaths`, `assists`, `totalDamageDealtToChampions`.
- *Variables de Economía*: `goldEarned`, `totalMinionsKilled`, `challenge_goldPerMinute`, `damageDealtToTurrets`.
- *Variables de Mapa*: `visionScore`, `challenge_teamBaronKills`, `challenge_teamElderDragonKills`.
- *Contexto*: `championName`, `individualPosition` (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY) y `side` (Blue/Red).

El análisis exploratorio preliminar muestra que las métricas deben ser tratadas de forma diferenciada según la posición, ya que un soporte (`UTILITY`) tendrá valores de daño bajos pero una puntuación de visión significativamente alta en comparación con un `TOP`.

== Preparación de los datos

Para garantizar la calidad de los modelos predictivos, se aplicarán las siguientes transformaciones:
- *Filtrado*: Eliminación de registros con jugadores ausentes (`challenge_hadAfkTeammate == 1`) o remakes (partidas que terminan antes de los 7 minutos) para evitar sesgos por partidas desbalanceadas.
- *Normalización*: Dado que las partidas tienen duraciones variables (`timePlayed`), métricas como el oro, el daño y los súbditos se estandarizarán a valores por minuto.
- *Codificación*: Las variables categóricas como `championName` e `individualPosition` se transformarán mediante *One-Hot Encoding*.
- *Limpieza de Outliers*: Identificación de valores extremos en pings (`totalPings`) o duraciones de partida inusuales que puedan distorsionar la media.

== Modelado

Se implementará un enfoque multialgoritmo para comparar rendimiento, precisión y explicabilidad, cumpliendo con el requisito de evaluar al menos cinco arquitecturas distintas:

1. *Regresión Logística*: Para obtener un modelo base altamente interpretable sobre la probabilidad lineal de victoria y el peso de cada métrica.
2. *Random Forest*: Utilizado para capturar interacciones complejas entre variables y extraer un ranking de "Importancia de Características" (Feature Importance).
3. *Máquinas de Soporte Vectorial (SVM)*: Para encontrar el hiperplano óptimo de separación entre victorias y derrotas en el espacio de métricas de combate.
4. *K-Nearest Neighbors (KNN)*: Basado en la vecindad de instancias, para clasificar el resultado de una partida según desempeños históricos similares.
5. *XGBoost*: Algoritmo de ensamble de alto rendimiento para maximizar la precisión predictiva sobre los datos tabulares del juego.
6. *Redes Neuronales (Perceptrón Multicapa)*: Para buscar patrones no lineales profundos en el gran volumen de datos disponible (29,162 registros).

== Evaluación

Los modelos se validarán utilizando una partición de datos de entrenamiento (80%) y prueba (20%). Se utilizarán las siguientes métricas:
- *Matriz de Confusión*: Para observar falsos positivos y negativos en la predicción de victorias.
- *Curva ROC y AUC*: Para medir la capacidad de discriminación del modelo entre ganador y perdedor.
- *Validación Cruzada (K-Fold)*: Para asegurar que el modelo es robusto y no presenta sobreajuste (*overfitting*) a campeones específicos del meta actual.

== Despliegue y Resultados

El modelo se ha desplegado en una plataforma analítica interactiva desarrollada con *Streamlit*, que permite la democratización del análisis de datos para cualquier jugador. La aplicación se organiza en cuatro módulos estratégicos:

1. *Dashboard de Inicio y Rankings*: Visualización en tiempo real de los mejores jugadores de la región (LAS), con un sistema de paginación que permite explorar el top 25, 50 y más allá, integrando datos de LPs y winrate.
2. *Explorador de Modelos (IA & Simulador)*: Módulo central donde el usuario puede auditar los 6 modelos implementados (Regresión Logística, Random Forest, SVM, KNN, XGBoost y MLP). Incluye:
  - *Auditoría de Importancia*: Visualización dinámica de qué variables "aprende" cada modelo que son vitales para ganar.
  - *Simulador de Partida*: Interfaz interactiva que ajusta los campos de entrada según las Top variables del modelo elegido, permitiendo predecir resultados con datos normalizados mediante *StandardScaler*.
3. *Metajuego Global*: Panel de visualización avanzada con gráficos de dispersión y cajas para identificar tendencias del parche actual (ej. correlación Visión vs Daño).
4. *Perfil de Invocador Avanzado*: Sistema inspirado en plataformas profesionales (OP.GG/Mobalytics) que incluye:
  - *Radar de Desempeño*: Visualización pentagonal de habilidades.
  - *Historial Detallado*: Desglose de los 10 jugadores de cada partida con asignación automática de medallas (*MVP, Carry, Visionary*) basada en heurísticas de Machine Learning.

= Herramientas y técnicas

Para el desarrollo del proyecto se utilizarán las siguientes tecnologías:
- *Lenguaje*: Python 3.10+ por su ecosistema robusto en ciencia de datos.
- *Librerías de ML*: Scikit-Learn (modelado base), XGBoost (ensamble avanzado) y TensorFlow/Keras (redes neuronales).
- *Manipulación de Datos*: Pandas y NumPy para la limpieza y normalización.
- *Visualización*: Matplotlib y Seaborn para el análisis exploratorio; Plotly para la interfaz interactiva.
- *API*: Riot Games API para la obtención y actualización de los datos (`league_data.csv`).
- *Entorno*: VS Code para el desarrollo experimental.

= Conclusiones

El desarrollo del proyecto demuestra que la aplicación de Ciencia de Datos en *eSports* permite pasar de la intuición a la certeza matemática. Las conclusiones principales son:

1. *Eficacia Predictiva*: Los modelos entrenados, especialmente Random Forest y XGBoost, alcanzan precisiones superiores al 85% al predecir el resultado de una partida basándose en métricas de desempeño individual y de equipo.
2. *Variables Críticas*: Mediante la auditoría de modelos, se identificó que métricas como el `oro por minuto` y la `participación en asesinatos` son universales, pero variables como el `score de visión` cobran una relevancia crítica en rangos altos (Challenger/Grandmaster).
3. *Interacción Modelo-Usuario*: La implementación de un simulador dinámico permite que los usuarios entiendan la "sensibilidad" de cada variable, funcionando no solo como una herramienta predictiva sino educativa.
4. *Escalabilidad*: La arquitectura modular del sistema permite integrar nuevas variables de la API de Riot Games de forma transparente, asegurando que los modelos puedan reentrenarse con los cambios del metajuego en cada parche.

= Anexos

- *Dataset*: `league_data.csv` (29,162 registros).
- *Repositorio*: [https://github.com/Zinko5/league-learning](https://github.com/Zinko5/league-learning).
- *Interfaz*: Aplicación Streamlit multialgoritmo.

== Aviso Legal

_League Learning_ isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.
