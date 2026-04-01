#import "caratulaPlantillaT.typ": *

#import "apuntesPlantillaT.typ": *
#show: config

#caratula(
  "Boceto 2",
  "DAT 255 - Machine Learning",
  "M. Sc. Menfy Morales Ríos",
  "Gabriel Marcelo Muñoz Callisaya",
  datetime(year: 2026, month: 03, day: 24),
)

#heading(text(size: 2em)[League Learning], outlined: false, numbering: none)

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

El método se divide en 6 fases:

1. Entendimiento del negocio
2. Entendimiento de los datos
3. Preparación de los datos
4. Modelado
5. Evaluación
6. Despliegue

= Desarrollo

== Entendimiento del negocio

El objetivo principal de este desarrollo es identificar y cuantificar los factores críticos que determinan la victoria en partidas competitivas de League of Legends. En un entorno de saturación de datos crudos donde la información accionable permanece oculta, se busca construir un sistema que permita a jugadores y analistas priorizar métricas de rendimiento que tengan una correlación directa con el éxito. El negocio del *coaching* y el análisis profesional se beneficia de modelos que no solo predigan el resultado, sino que expliquen qué variables (visión, control de objetivos, ventaja en fase de líneas) deben optimizarse para mejorar el *winrate*.

== Entendimiento de los datos

El dataset base, `league_data.csv`, consta de aproximadamente 42,000 registros capturados a través de la API de Riot Games. Cada registro representa el desempeño individual de un jugador en una partida específica, utilizando un conjunto de 18 variables clave para el modelado:

- *Variable Objetivo*: `win` (booleano), que indica si el jugador ganó o perdió la partida.
- *Métricas Base (Normalizadas por minuto)*: `goldEarned`, `totalMinionsKilled`, `totalDamageDealtToChampions`, `totalDamageTaken`, `damageDealtToEpicMonsters`, `damageDealtToTurrets`, `kills`, `deaths`, `assists`, `visionScore`.
- *Métricas de Desafío y Equipo (Acumulativas)*: `challenge_teamRiftHeraldKills`, `challenge_teamBaronKills`, `challenge_teamElderDragonKills`, `challenge_highestChampionDamage`, `challenge_killParticipation`, `challenge_laningPhaseGoldExpAdvantage`, `challenge_teamDamagePercentage`, `totalPings`.
- *Contexto de Segmentación*: `individualPosition` (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY), utilizada para entrenar modelos especializados por rol.

El análisis exploratorio preliminar muestra que las métricas deben ser tratadas de forma diferenciada según la posición, ya que un soporte (`UTILITY`) tendrá valores de daño bajos pero una puntuación de visión significativamente alta en comparación con un `TOP`.

Para una gestión profesional, los datos se organizan en un directorio `data/` independiente del código, permitiendo un flujo de trabajo de *Data Version Control (DVC)* que asegura la trazabilidad entre el código fuente y los datasets pesados.

== Preparación de los datos

Para garantizar la calidad de los modelos predictivos, se aplican las siguientes transformaciones:
- *Filtrado*: Eliminación de registros con jugadores ausentes (`challenge_hadAfkTeammate == 1`) o partidas de duración extremadamente corta (menos de 7 minutos) para evitar sesgos por partidas desbalanceadas o incompletas.
- *Normalización Per-Minute*: Dado que las partidas tienen duraciones variables (`timePlayed`), las métricas base se estandarizan a valores por minuto para permitir una comparación justa entre partidas rápidas y largas.
- *Estandarización de Escala*: Uso de `StandardScaler` para centrar y escalar todas las características numéricas, asegurando que algoritmos sensibles a la magnitud (como KNN o MLP) operen de forma óptima.
- *Segmentación por Rol*: En lugar de utilizar One-Hot Encoding para la posición, se opta por una arquitectura de modelos especializados donde cada rol posee su propio conjunto de 9 algoritmos entrenados exclusivamente con datos de esa posición.

== Modelado

Se implementa un enfoque multialgoritmo integrando 9 arquitecturas distintas: Regresión Logística, Random Forest, XGBoost, KNN, LDA, Naive Bayes, Árbol de Decisión, SVM y Redes Neuronales (MLP). El modelado se automatiza mediante un pipeline que asegura que cada configuración de entrenamiento se registre como un experimento único.

== Evaluación

Los modelos se validan utilizando una partición de datos de prueba (10%). Se calculan métricas de ROC-AUC, Accuracy y F1-Score. Se jerarquizan las variables mediante *Permutation Importance*, permitiendo una explicabilidad unificada de los factores influyentes. Para asegurar la calidad continua, se ha integrado *MLflow*, que permite auditar cada versión de entrenamiento, comparando el rendimiento a través de diferentes "Runs" o ejecuciones históricas.

== Despliegue y Resultados

El sistema se despliega mediante una aplicación interactiva centrada en el usuario, incorporando principios de *MLOps* para gestionar el ciclo de vida del proyecto:

1. *Dashboard de Inicio*: Visualización de rankings regionales LAS.
2. *IA & Simulador Multialgoritmo*: Permite predecir resultados ajustando variables tácticas.
3. *Gestión de Versiones (MLOps)*: El despliegue incluye un selector de versiones de Inteligencia Artificial que consulta los experimentos registrados en *MLflow*. Esto permite al usuario "viajar en el tiempo" entre diferentes versiones de entrenamiento y datasets capturados.
4. *Historial y Radar de Desempeño*: Diagnóstico individualizado del nivel de juego.

La arquitectura se apoya en enlaces simbólicos y una estructura modular de carpetas (`data/`, `modelos/`, `metricas/`) para asegurar que el despliegue sea robusto y escalable ante el dinámico metajuego de League of Legends.

= Herramientas y técnicas

Para el desarrollo del proyecto se utilizarán las siguientes tecnologías:
- *Lenguaje*: Python 3.10+ por su ecosistema robusto en ciencia de datos.
- *Librerías de ML*: Scikit-Learn, XGBoost y TensorFlow/Keras.
- *Gestión de Experimentos*: MLflow para el rastreo de ejecuciones, métricas y registro de modelos.
- *Versionado de Datos*: DVC para la persistencia y trazabilidad de los datasets físicos.
- *Manipulación de Datos*: Pandas y NumPy para la limpieza y normalización.
- *Visualización*: Plotly para la interfaz interactiva.
- *API*: Riot Games API para la obtención y actualización de los datos.

= Conclusiones

1. *Eficacia Predictiva*: Los modelos alcanzan precisiones superiores al 85% al predecir el resultado basándose en métricas de desempeño.
2. *Trazabilidad MLOps*: La implementación de MLflow permite una auditoría profesional, asegurando que cada predicción esté vinculada a una versión específica de datos y modelos.
3. *Impacto Estratégico*: El simulador dinámico permite entender la sensibilidad de las variables, facilitando la toma de decisiones basada en evidencia matemática.

= Anexos

== Repositorio de Código Abierto

El código fuente del proyecto está disponible en el siguiente repositorio de GitHub:

- *Repositorio*: #link("https://github.com/Zinko5/league-learning")[League Learning]\
(https://github.com/Zinko5/league-learning).

== Aviso Legal

Debido a que el proyecto utiliza datos obtenidos a través de la API de Riot Games, se debe incluir el siguiente aviso legal:

_League Learning_ isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.
