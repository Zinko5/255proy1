// #import "/utiles/typst/plantillas/caratulaPlantillaT.typ": *

// #import "/utiles/typst/plantillas/apuntesPlantillaT.typ": *
// #show: config

// #caratula(
//   "Boceto 2",
//   "DAT 255",
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

El gran volumen de datos generado por las partidas de League of Legends impone una carga analítica que satura el procesamiento manual de patrones de rendimiento. Esta complejidad inherente al juego propicia una interpretación fragmentada sobre su comportamiento táctico y estratégico. En este contexto, el uso de herramientas de procesamiento sistemático permite transformar este volumen masivo en inteligencia accionable.

== Árbol de problema

- *Problema Central*: Opacidad de los factores determinantes de victoria ante el volumen masivo de datos no procesados.
- *Causas*:
  - *Saturación de Datos*: Presencia de variables crudas en el dataset que sobrepasan la capacidad del análisis estadístico convencional.
  - *Análisis Descriptivo Superficial*: Prevalencia de métodos rudimentarios que restringen el procesamiento a estadísticas básicas, bloqueando la detección de patrones complejos.
  - *Hermetismo de la Información*: Predominancia de resultados técnicos de Machine Learning inaccesibles para el usuario sin perfil analítico.
- *Efectos*:
  - *Dependencia de la Intuición*: Toma de decisiones anclada en percepciones subjetivas por encima de la evidencia cuantitativa.
  - *Predicciones Volátiles*: Generación de pronósticos lentos y con bajo rigor matemático en entornos competitivos.
  - *Marginalización de métricas estratégicas*: Concentración excesiva en estadísticas básicas (KDA) ignorando variables determinantes como visión y utilidad.

#image("arbol_problema_final.png")

= Objetivo

El objetivo de este proyecto es transformar datos brutos de partidas de League of Legends en inteligencia accionable mediante el desarrollo de una plataforma analítica. Utilizando un enfoque multialgoritmo de Machine Learning (Modelos Lineales, Ensambles, SVM, KNN y Redes Neuronales), se busca identificar métricas críticas y generar modelos predictivos personalizados que optimicen la toma de decisiones tácticas de los jugadores.

== Árbol de objetivos

- *Objetivo Central*: Desarrollar una plataforma analítica basada en Machine Learning que identifique patrones de victoria y proporcione modelos predictivos personalizados para jugadores de League of Legends.

- *Medios*:
  - *Procesamiento de Datos*: Implementar técnicas de minería de datos para limpiar y normalizar las variables de `league_data.csv`.
  - *Modelado Multialgoritmo*: Implementar y evaluar una arquitectura competitiva que incluya modelos de Regresión Logística, Random Forest, SVM, KNN, XGBoost y Redes Neuronales para garantizar la máxima precisión y capacidad predictiva.
  - *Personalización Algorítmica*: Crear un motor que permita aplicar estos modelos a los datos específicos de un jugador (análisis individual vs. global).
  - *Desarrollo Web*: Diseñar una interfaz interactiva (similar a op.gg pero profunda) para visualizar métricas clave y predicciones.
- *Fines*:
  - *Reducción de la Incertidumbre*: Determinación matemática de las métricas que garantizan un resultado favorable.
  - *Capacidad Predictiva*: Generación de pronósticos asertivos sobre el resultado de partidas basados en datos históricos y recientes.
  - *Optimización del Rendimiento*: Mejora del nivel de juego de los usuarios mediante el entendimiento de variables "ocultas" o poco utilizadas.

#image("arbol_objetivo_final.png")

= Justificación

El análisis de datos en el contexto de los videojuegos ha ganado popularidad en los últimos años, ya que permite a los desarrolladores y jugadores entender mejor el comportamiento del juego y mejorar el rendimiento de los jugadores. En el caso de League of Legends, diversas páginas web y herramientas han surgido para analizar datos de partidas, como op.gg, u.gg, leagueofgraphs.com, entre otras. Estas herramientas se mantienen en el análisis descriptivo sin proyectar tendencias predictivas ni estudios de profundidad algorítmica. Ante este predominio de la información histórica, el desarrollo de una plataforma que trascienda hacia el análisis predictivo es un proyecto ambicioso que optimiza el entendimiento competitivo.

En el contexto competitivo profesional, el análisis de datos es aún más importante, ya que permite a los equipos y jugadores identificar patrones que puedan mejorar su rendimiento. Por ejemplo, un equipo puede analizar sus partidas y darse cuenta de que tiene un mayor porcentaje de victorias cuando juega con un determinado campeón, o cuando juega en un determinado orden. Esta información puede ser utilizada para mejorar su estrategia y aumentar sus posibilidades de ganar.

= Metodología

El método seleccionado para el desarrollo es CRISP-DM, ya que permite un análisis sistemático y riguroso de los datos, lo que facilita la identificación de patrones y la construcción de modelos predictivos.

= Desarrollo

== Entendimiento del negocio

El objetivo principal de este desarrollo es identificar y cuantificar los factores críticos que determinan la victoria en partidas competitivas de League of Legends. En un entorno de saturación de datos crudos donde la información accionable permanece oculta, se busca construir un sistema que permita a jugadores y analistas priorizar métricas de rendimiento que tengan una correlación directa con el éxito. El negocio del *coaching* y el análisis profesional se beneficia de modelos que no solo predigan el resultado, sino que expliquen qué variables (visión, control de objetivos, ventaja en fase de líneas) deben optimizarse para mejorar el *winrate*.

== Entendimiento de los datos

El dataset base, `league_data.csv`, consta de 29,162 registros capturados a través de la API de Riot Games. Cada registro representa el desempeño individual de un jugador en una partida específica, abarcando 30 variables clave definidas en el `diccionario.md`:

- *Variable Objetivo*: `win` (booleano), que indica si el jugador ganó o perdió la partida.
- *Variables de Combate*: `kills`, `deaths`, `assists`, `totalDamageDealtToChampions`.
- *Variables de Economía*: `goldEarned`, `challenge_goldPerMinute`, `damageDealtToTurrets`.
- *Variables de Mapa*: `visionScore`, `challenge_teamBaronKills`, `challenge_teamElderDragonKills`.
- *Contexto*: `championName`, `individualPosition` (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY) y `side` (Blue/Red).

El análisis exploratorio preliminar muestra que las métricas deben ser tratadas de forma diferenciada según la posición, ya que un soporte (`UTILITY`) tendrá valores de daño bajos pero una puntuación de visión significativamente alta en comparación con un `TOP`.

== Preparación de los datos

Para garantizar la calidad de los modelos predictivos, se aplicarán las siguientes transformaciones:
- *Filtrado*: Eliminación de registros con jugadores ausentes (`challenge_hadAfkTeammate == 1`) o remakes (partidas que terminan antes de los 7 minutos) para evitar sesgos por partidas desbalanceadas.
- *Normalización*: Dado que las partidas tienen duraciones variables (`timePlayed`), métricas como el oro y el daño se estandarizarán a valores por minuto.
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
