#import "/utiles/typst/plantillas/caratulaPlantillaT.typ": *

#import "/utiles/typst/plantillas/apuntesPlantillaT.typ": *
#show: config

#caratula(
  "Boceto 2",
  "DAT 255",
  "M. Sc. Menfy Morales Ríos",
  "Gabriel Marcelo Muñoz Callisaya",
  datetime(year: 2026, month: 03, day: 24),
)

#outline()

#pagebreak()

= Introducción

Análisis de datos de partidas de League of Legends, para realizar predicciones de distintas variables del juego y así poder entender mejor el comportamiento del juego. Útil para jugadores, equipos y analistas que quieran mejorar su rendimiento en el juego.

= Antecedentes

League of Legends es un juego multijugador en línea desarrollado y publicado por Riot Games. Es un juego MOBA (Multiplayer Online Battle Arena) en el que dos equipos de cinco jugadores compiten entre sí para destruir la base del equipo contrario. El juego fue lanzado en 2009 y desde entonces se ha convertido en uno de los juegos más populares del mundo, con millones de jugadores en todo el mundo.

Con una popularidad que se mantiene a lo largo de los años, el juego ha desarrollado un ecosistema competitivo profesional exitoso, con ligas profesionales en todo el mundo y un campeonato mundial que atrae a decenas de miles de espectadores, el juego se ha convertido en la principal fuente de ingresos para varios jugadores profesionales y sus equipos de entrenadores y analistas.

= Problemática

El gran volumen de datos generados por las partidas de League of Legends hace que sea difícil para los jugadores y equipos analizar manualmente el rendimiento de sus partidas y identificar patrones que puedan mejorar su rendimiento. Además, la complejidad del juego hace que sea difícil para los jugadores entender completamente el comportamiento del juego y cómo mejorar su rendimiento. Por lo tanto, se hace necesario el uso de herramientas que permitan analizar estos datos de manera eficiente y obtener información valiosa que pueda ser utilizada para mejorar el rendimiento de los jugadores.

== Árbol de problema

- *Problema Central*: Dificultad para identificar factores determinantes de victoria en League of Legends debido a la alta dimensionalidad y volumen de datos no procesados.
- *Causas*:
  - *Complejidad del Dato*: Exceso de variables crudas en el dataset que superan la capacidad de análisis estadístico convencional.
  - *Inexistencia de Herramientas*: Ausencia de modelos predictivos y algoritmos de clasificación que den sentido lógico a los datos.
  - *Barrera de Acceso*: Falta de una interfaz que traduzca resultados complejos de Machine Learning en información accionable para el usuario común.
- *Efectos*:
  - Toma de decisiones basada en intuición y no en evidencia cuantitativa.
  - Incapacidad de predecir resultados de forma asertiva y rápida.
  - Desaprovechamiento de métricas no convencionales (visión, utilidad, timing) que influyen en el winrate.

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

El análisis de datos en el contexto de los videojuegos ha ganado popularidad en los últimos años, ya que permite a los desarrolladores y jugadores entender mejor el comportamiento del juego y mejorar el rendimiento de los jugadores. En el caso de League of Legends, diversas páginas web y herramientas han surgido para analizar datos de partidas, como op.gg, u.gg, leagueofgraphs.com, entre otras. Estas herramientas permiten a los jugadores ver estadísticas de sus partidas, analizar el rendimiento de sus campeones, comparar sus estadísticas con las de otros jugadores, etc. pero se limitan a mostrar estadísticas y no realizan predicciones ni análisis profundos. Por lo tanto, se hace necesario el uso de herramientas que permitan analizar estos datos de manera eficiente y obtener información valiosa que pueda ser utilizada para mejorar el rendimiento de los jugadores. El desarrollo de una plataforma analítica que permita pasar del análisis descriptivo al análisis predictivo es un proyecto ambicioso pero necesario.

En el contexto competitivo profesional, el análisis de datos es aún más importante, ya que permite a los equipos y jugadores identificar patrones que puedan mejorar su rendimiento. Por ejemplo, un equipo puede analizar sus partidas y darse cuenta de que tiene un mayor porcentaje de victorias cuando juega con un determinado campeón, o cuando juega en un determinado orden. Esta información puede ser utilizada para mejorar su estrategia y aumentar sus posibilidades de ganar.

= Metodología

El método seleccionado para el desarrollo es CRISP-DM, ya que permite un análisis sistemático y riguroso de los datos, lo que facilita la identificación de patrones y la construcción de modelos predictivos.

= Desarrollo

== Entendimiento del negocio

El objetivo principal de este desarrollo es identificar y cuantificar los factores críticos que determinan la victoria en partidas competitivas de League of Legends. En un entorno donde los datos crudos abundan pero la información accionable es escasa, se busca construir un sistema que permita a jugadores y analistas priorizar métricas de rendimiento que tengan una correlación directa con el éxito. El negocio del *coaching* y el análisis profesional se beneficia de modelos que no solo predigan el resultado, sino que expliquen qué variables (visión, control de objetivos, ventaja en fase de líneas) deben optimizarse para mejorar el *winrate*.

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

== Despliegue

El modelo final se integrará en una plataforma analítica interactiva. La arquitectura contempla:
- Una interfaz donde el usuario pueda cargar su historial de partidas.
- Un panel de "Diagnóstico de Victoria" que compare sus métricas actuales con el perfil ideal generado por el modelo.
- Recomendaciones automáticas basadas en las debilidades detectadas (ej. "Tu probabilidad de victoria subiría un 8% si aumentaras tu participación en objetivos globales en el juego temprano").

= Herramientas y técnicas

Para el desarrollo del proyecto se utilizarán las siguientes tecnologías:
- *Lenguaje*: Python 3.10+ por su ecosistema robusto en ciencia de datos.
- *Librerías de ML*: Scikit-Learn (modelado base), XGBoost (ensamble avanzado) y TensorFlow/Keras (redes neuronales).
- *Manipulación de Datos*: Pandas y NumPy para la limpieza y normalización.
- *Visualización*: Matplotlib y Seaborn para el análisis exploratorio; Plotly para la interfaz interactiva.
- *API*: Riot Games API para la obtención y actualización de los datos (`league_data.csv`).
- *Entorno*: VS Code para el desarrollo experimental.

= Conclusiones

= Anexos
