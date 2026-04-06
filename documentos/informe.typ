#import "caratulaPlantillaT.typ": *

#import "apuntesPlantillaT.typ": *
#show: config

#caratula(
  "Aprendizaje Supervisado - League Learning",
  "DAT 255 - Machine Learning",
  "M. Sc. Menfy Morales Ríos",
  "Gabriel Marcelo Muñoz Callisaya",
  datetime(year: 2026, month: 04, day: 07),
)

#heading(text(size: 1.6em)[League Learning], outlined: false, numbering: none)

#outline()

#pagebreak()

= Introducción

En el ecosistema altamente competitivo de League of Legends (LoL), la victoria no solo depende de la ejecución mecánica, sino de una toma de decisiones estratégica fundamentada en la gestión eficiente de recursos, visión y objetivos tácticos. Sin embargo, los jugadores y analistas actuales se enfrentan a una "paradoja del dato": cada partida genera una densidad masiva de indicadores técnicos que, en su estado bruto, resultan extremadamente difíciles de jerarquizar. El presente proyecto, denominado *League Learning*, surge como una solución técnica integral para transformar este flujo masivo de datos en conocimiento predictivo y modelos de decisión accionables.

La esencia de este trabajo radica en cerrar la brecha entre la intuición subjetiva y la evidencia matemática mediante el uso de *Modelos de Aprendizaje Supervisado*. A diferencia de las herramientas de análisis convencionales que operan a nivel descriptivo, *League Learning* implementa un motor de inferencia multialgoritmo que integra 9 arquitecturas distintas: desde modelos lineales y probabilísticos (LR, LDA, NB) hasta modelos de alta complejidad como XGBoost, Random Forest y Redes Neuronales (MLP). Este enfoque permite no solo predecir el resultado de una partida con una precisión superior al 85%, sino también extraer la importancia de las variables (*Feature Importance*) mediante técnicas de permutación, unificando la explicabilidad incluso en modelos de "caja negra".

Para garantizar la solidez del desarrollo, se ha seguido rigurosamente la metodología *CRISP-DM* (*Cross-Industry Standard Process for Data Mining*), asegurando una transición lógica entre el entendimiento del negocio de los eSports y la preparación técnica de los datos (normalización per-minute, filtrado de AFKs y segmentación por rol). Un pilar fundamental del proyecto es la incorporación de principios de *MLOps* a través de la herramienta *MLflow*, permitiendo una trazabilidad total del ciclo de vida del modelo, desde la captura del dataset en la API de Riot Games hasta el despliegue de versiones históricas que el usuario puede consultar interactivamente.

El resultado final plasmado en este documento y en la aplicación desplegada es una plataforma que ofrece:
- *Rankings Globales*: Contextualización del rendimiento en entornos competitivos reales.
- *Simulador Dinámico de Victoria*: Una interfaz interactiva que permite ajustar variables tácticas para observar su impacto probabilístico en tiempo real.
- *Diagnóstico Especializado por Rol*: Modelos entrenados independientemente para cada posición (TOP, JUNGLE, MID, ADC, SUPP), reconociendo que los factores de éxito varían drásticamente según la función del jugador.
- *Gestión de Versiones de IA*: Un sistema de "viaje en el tiempo" que permite auditar cómo evolucionan las predicciones según el dataset de entrenamiento y la arquitectura seleccionada.

Con este enfoque, *League Learning* se posiciona no solo como una herramienta de predicción, sino como un marco de referencia estratégico que fundamenta la planificación del entrenamiento y la toma de decisiones en el gaming profesional bajo un rigor puramente científico.

= Antecedentes

League of Legends (LoL), desarrollado por Riot Games en 2009, se ha consolidado como el referente global en el género *Multiplayer Online Battle Arena* (MOBA). Su estructura competitiva, que enfrenta a dos equipos de cinco jugadores en el mapa de la Grieta del Invocador, demanda una coordinación técnica y estratégica de nivel profesional. Con el tiempo, este ecosistema ha evolucionado de simples partidas recreativas a una industria millonaria donde el análisis táctico sofisticado es un requisito básico para la competitividad en los mejores servidores del mundo.

Históricamente, la planificación estratégica en LoL dependía casi exclusivamente de la experiencia y la intuición de los jugadores veteranos. Las herramientas de análisis tempranas se limitaban a registrar valores agregados de final de partida, como el daño total o el oro acumulado, que a menudo resultaban insuficientes para entender la dinámica real del juego. Sin embargo, la introducción de la *API de Riot Games* marcó un punto de inflexión, permitiendo la captura de registros altamente granulares (como los procesados en el dataset base de este proyecto) que incluyen métricas avanzadas de control de visión, participación en objetivos épicos y ventajas en fase de líneas.

Este acceso masivo a la información ha generado una "inflación de datos", donde el exceso de métricas crudas a menudo oculta la importancia relativa de cada acción. Un antecedente crítico para el desarrollo de *League Learning* es el reconocimiento de que las estadísticas convencionales fallan al no considerar la naturaleza no lineal del juego y la especialización por rol. Por ejemplo, el éxito de un soporte (*Utility*) no se mide con las mismas variables que el de un carrilero central (*Middle*), lo que genera la necesidad de modelos capaces de segmentar y jerarquizar los factores de victoria de forma independiente. Esta brecha entre la disponibilidad de datos y la capacidad de extracción de conocimiento predictivo fundamenta la transición hacia el *Aprendizaje Supervisado*.

= Problemática

En el entorno competitivo de League of Legends, la planificación estratégica y técnica de los jugadores se sustenta de forma predominante en la intuición individual debido a la alta densidad de indicadores técnicos generados en cada partida. La multiplicidad de variables de desempeño —que abarcan desde la captura minuciosa de la economía y el control de visión hasta complejos indicadores de combate y participación en objetivos— genera un entorno de *alta dimensionalidad* que excede la capacidad de procesamiento manual y el análisis estadístico convencional. Los jugadores y cuerpos técnicos se encuentran ante un flujo incesante de registros brutos donde resulta estructuralmente imposible determinar, sin herramientas de inteligencia computacional, qué métricas específicas tienen un impacto causal real en el resultado y cuáles son meramente correlativas o incidentales.

Esta dificultad para sistematizar y jerarquizar los registros tácticos impide que el jugador cuantifique con precisión el impacto de sus acciones sobre la probabilidad de victoria. Como consecuencia, el diagnóstico de rendimiento se construye habitualmente sobre percepciones parciales o sesgos de confirmación, donde a menudo se sobrevalora el desempeño en combate ignorando factores críticos de control de mapa o eficiencia en la fase de líneas. Las herramientas de análisis actuales, al operar casi exclusivamente con promedios descriptivos y valores agregados, no ofrecen una visión de *sensibilidad*: el usuario carece de un marco que le indique exactamente qué variables debe optimizar prioritariamente para mejorar su porcentaje de éxito de manera medible.

Además, la problemática se profundiza por la ausencia de una analítica diferenciada por rol en los diagnósticos estándar. Una estadística que sea determinante para un carrilero superior (*Top*) puede ser irrelevante para un soporte (*Utility*), lo que genera una distorsión en la evaluación del rendimiento individual. Esta carencia de un sistema que aplique *Aprendizaje Supervisado* para segmentar y priorizar métricas críticas condena a los jugadores a una planificación táctica de "ensayo y error", limitando su crecimiento profesional por la falta de una arquitectura de datos que traduzca la complejidad del juego en recomendaciones estratégicas fidedignas.

== Árbol de problema

- *Problema Central*: Los jugadores analizan de forma manual los factores determinantes de la victoria en League of Legends a través de estadísticas descriptivas.
// - *Problema Central*: Identificación de factores determinantes de la victoria mediante revisión manual de estadísticas descriptivas en League of Legends.
- *Causas*:
  - Presentación de indicadores técnicos en valores agregados y aislados por categoría en las pantallas de resumen de partida.
  - Estructura de los datos de rendimiento con múltiples variables simultáneas en cada partida.
  - Almacenamiento de trayectorias históricas de partidas como datos brutos en las bases de datos accesibles.
- *Efectos*:
  - Selección de áreas de práctica basada en la prominencia de métricas en revisiones individuales de partidas.
  - Evaluación del rendimiento centrada en métricas aisladas por categoría.
  - Utilización de registros históricos como datos estáticos para la autoevaluación competitiva.

#image("img/arbol_problema_final.png")

= Objetivo

Optimizar el diagnóstico del rendimiento táctico mediante el desarrollo de una plataforma analítica integral que sistematice, valide y jerarquice los factores determinantes de la victoria en League of Legends. El proyecto busca establecer un nuevo marco de referencia basado en evidencia matemática para la planificación estratégica, permitiendo que el jugador identifique con precisión quirúrgica los indicadores críticos que influyen en su desempeño competitivo a través de modelos robustos de *Machine Learning* y una arquitectura de *MLOps* que garantice la trazabilidad de los hallazgos científicos.

Para alcanzar esta optimización, el objetivo se centra en la construcción de un sistema capaz de ingerir y procesar datos granulares de la *API de Riot Games*, transformándolos en métricas normalizadas (*per-minute*) que aseguren una base de comparación equitativa entre partidas de diversa duración. El núcleo técnico del proyecto reside en la implementación de una arquitectura multialgoritmo que no solo prediga el resultado de la partida con alta fidelidad, sino que actúe como un motor de *diagnóstico de sensibilidad*, revelando al usuario qué variables tácticas específicas —desde el control de visión hasta la eficiencia económica en la fase de líneas— requieren una intervención prioritaria para mejorar su rendimiento competitivo.

Finalmente, el objetivo trasciende el análisis estadístico convencional al integrar un *Simulador Dinámico de Victoria*. Esta herramienta interactiva democratiza el acceso a análisis de alto rendimiento —típicamente reservados para equipos profesionales con analistas técnicos dedicados— permitiendo al usuario interactuar directamente con los modelos de inteligencia artificial para observar cómo fluctúa la probabilidad de éxito al ajustar sus propios parámetros de juego. Con este enfoque, *League Learning* busca convertir datos inmanejables en recomendaciones estratégicas claras, fidedignas y accionables para la profesionalización del entrenamiento.

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

#image("img/arbol_objetivo_final.png")

= Justificación

El análisis de datos en el contexto de los eSports ha evolucionado hacia un componente crítico para la optimización del rendimiento estratégico. En League of Legends, aunque existen diversas plataformas de consulta de estadísticas, estas operan predominantemente en un nivel descriptivo, mostrando resultados históricos sin proyectar tendencias predictivas ni establecer la importancia relativa de cada acción sobre el resultado final. Ante esta limitación, el desarrollo de un sistema que trascienda hacia el análisis predictivo y la jerarquización de variables permite que el jugador abandone la dependencia de la intuición subjetiva y adopte una estrategia basada en evidencia matemática rigurosa.

La justificación técnica y el valor diferencial de este proyecto se sustentan en tres pilares de innovación integrados en la arquitectura de *League Learning*:

1. *Sistema Multialgorítmico*: A diferencia de implementar un modelo aislado, el sistema evalúa simultáneamente 9 arquitecturas distintas (desde funciones lineales como LR y LDA hasta estructuras complejas como XGBoost, SVM y Redes Neuronales MLP). Esta diversidad es fundamental porque la naturaleza táctica del juego varía drásticamente según el rol; mientras que un árbol de decisión puede capturar la lógica binaria de la economía en un carril, patrones no lineales y dispersos en otras posiciones requieren la potencia de aprendizaje de una red neuronal. El enfoque multialgorítmico permite validar la consistencia de las predicciones y seleccionar dinámicamente el modelo con mejor desempeño (*ROC-AUC / F1-Score*) para cada escenario competitivo.

2. *Explicabilidad Unificada (Permutation Importance)*: Uno de los mayores desafíos en la Inteligencia Artificial es el problema de la "caja negra". Este proyecto justifica el uso de *Permutation Importance* para estandarizar la interpretación de todos los modelos bajo una misma métrica de sensibilidad. Esta técnica permite medir el peso específico de variables críticas (como el *visionScore_perMin* o la *laningPhaseGoldExpAdvantage*) de forma agnóstica al modelo, garantizando que el diagnóstico entregado al jugador sea transparente y accionable, sin importar la complejidad del algoritmo que lo sustente.

  El *Permutation Importance* se calcula de la siguiente manera:
  - Se calcula la métrica de desempeño del modelo original.
  - Se permuta aleatoriamente una variable de entrada.
  - Se vuelve a calcular la métrica de desempeño del modelo con la variable permutada.
  - La diferencia entre la métrica original y la métrica con la variable permutada es el *Permutation Importance* de esa variable.

3. *Gestión Profesional MLOps (MLflow)*: En un entorno dinámico donde el balance del juego cambia periódicamente, la reproducibilidad y el versionado son esenciales para la integridad de los datos. La integración con *MLflow* asegura que cada ejecución de entrenamiento, métrica de validación y snapshot del dataset capturado desde la API de Riot Games esté debidamente auditado. Esto permite un flujo de trabajo de *MLOps* real, donde el usuario final de la plataforma puede consultar versiones históricas de la "Inteligencia de la IA", comparando el rendimiento y asegurando una evolución del sistema basada en auditorías técnicas fidedignas.

En el ámbito profesional, la capacidad de identificar patrones ocultos representa una ventaja competitiva decisiva. La implementación de este motor técnico no solo democratiza el acceso a diagnósticos de alta fidelidad, sino que posiciona a *League Learning* como una herramienta de transición obligatoria entre la observación pasiva de estadísticas y la ejecución estratégica fundamentada en la ciencia de datos predictiva.

= Metodología

Para el desarrollo sistemático del proyecto *League Learning*, se ha seleccionado la metodología *CRISP-DM* (*Cross-Industry Standard Process for Data Mining*). Este estándar de la industria proporciona un marco de trabajo cíclico, iterativo y altamente estructurado, ideal para proyectos de Ciencia de Datos donde la comprensión del dominio del negocio y la calidad del preprocesamiento son factores críticos de éxito. La naturaleza dinámica de League of Legends —un entorno con parches de actualización constantes que alteran el equilibrio tensional— requiere un enfoque que permita re-evaluar y re-entrenar modelos de forma rigurosa.

El proceso se articula a través de las siguientes 6 fases, adaptadas específicamente a los desafíos del análisis competitivo de eSports:

1. *Entendimiento del negocio*: En esta fase inicial se definen los objetivos estratégicos desde la perspectiva del entrenamiento profesional. El "negocio" se interpreta como la optimización del rendimiento táctico, buscando identificar qué indicadores específicos influyen realmente en la probabilidad de victoria para jugadores, entrenadores y analistas de datos.

2. *Entendimiento de los datos*: Consistente en la captura, auditoría y exploración inicial de los registros históricos obtenidos a través de la *API de Riot Games*. Se analizan las 18 variables clave y sus correlaciones iniciales con la variable objetivo `win`, detectando la necesidad de segmentar la información por posición competitiva.

3. *Preparación de los datos*: Es la fase más intensiva del proyecto. Se aplican técnicas de *Ingeniería de Atributos*, destacando la normalización *per-minute* para permitir comparaciones justas entre partidas de diversa duración, el filtrado de registros sesgados (eliminación de jugadores AFK y partidas excesivamente cortas) y la estandarización de escalas mediante *StandardScaler* para los modelos sensibles a la magnitud.

4. *Modelado*: Implementación de un flujo de trabajo multialgoritmo que evalúa 9 arquitecturas distintas. Se realiza el ajuste de hiperparámetros (como la optimización dinámica mediante *GridSearch* para el algoritmo KNN) y el entrenamiento de modelos especializados para cada uno de los 5 roles del juego, asegurando una personalización total del diagnóstico.

5. *Evaluación*: Validación técnica de los modelos utilizando métricas de *ROC-AUC, Accuracy y F1-Score*. En esta etapa se integra el análisis de *Permutation Importance* para unificar la explicabilidad y validar que los patrones detectados por la IA coincidan con la lógica táctica del videojuego, evitando falsas correlaciones.

6. *Despliegue*: Integración de los modelos entrenados en una plataforma interactiva basada en *Streamlit*. Se incorporan prácticas de *MLOps* mediante *MLflow* para gestionar el ciclo de vida del proyecto, asegurando que el despliegue incluya versionado de modelos, trazabilidad de datasets y la capacidad de realizar auditorías de desempeño en tiempo real.

= Desarrollo

== Entendimiento del negocio

El objetivo estratégico de este desarrollo es identificar, cuantificar y jerarquizar los factores críticos que determinan la victoria en partidas competitivas de alto nivel en League of Legends. En el contexto de los eSports profesionales, el éxito se define por la capacidad de los equipos para optimizar sus ciclos de entrenamiento y toma de decisiones tácticas bajo un margen de error mínimo. Actualmente, este sector enfrenta una saturación de registros crudos donde la información verdaderamente accionable permanece oculta tras estadísticas agregadas, forzando a los cuerpos técnicos a depender excesivamente de la intuición subjetiva.

*League Learning* responde a esta necesidad mediante una plataforma de inteligencia competitiva que permite a jugadores, entrenadores y analistas priorizar métricas de rendimiento con una correlación directa y validada matemáticamente con el éxito. El valor de negocio del proyecto reside en su capacidad para transformar el análisis deportivo tradicional en un proceso de *Planificación Estratégica Basada en Evidencia*. Al integrar la predicción probabilística con el análisis de importancia de variables (*Feature Importance*), el sistema permite a una organización técnica identificar si su prioridad táctica inmediata debe centrarse, por ejemplo, en la optimización de la visión (*visionScore*) o en la maximización de la ventaja de recursos en la fase inicial del juego (*laningPhaseGoldExpAdvantage*).

Desde una perspectiva operativa, el sistema aporta valor en tres dimensiones críticas para el rendimiento competitivo:
- *Optimización del Coaching*: Provee a los entrenadores un marco de referencia fidedigno para fundamentar el *feedback* técnico, sustituyendo la observación anecdótica de partidas por diagnósticos multialgoritmo robustos.
- *Simulación Prospectiva*: A través del simulador dinámico integrado en la aplicación, el analista puede ejecutar escenarios "what-if" (p. ej., "¿Cómo variaría nuestra probabilidad de victoria si incrementamos la participación en muertes en un 15 %?"), facilitando la planificación táctica antes de la ejecución competitiva.
- *Adaptabilidad al Metajuego*: Dado que las mecánicas del juego sufren alteraciones periódicas por parches de balance, la infraestructura de *MLOps* asegura que el sistema pueda recalibrar sus modelos rápidamente, manteniendo la vigencia del diagnóstico táctico conforme evoluciona el panorama global de los mejores servidores.

Con este enfoque, el proyecto trasciende el análisis descriptivo convencional para constituirse como un motor de inteligencia que permite a los usuarios maximizar su *winrate* mediante el conocimiento profundo de las variables críticas que realmente deciden el resultado de una partida profesional.

== Entendimiento de los datos

El dataset base, `league_data.csv`, constituye el núcleo empírico del proyecto y consta de aproximadamente 42,000 registros capturados mediante la *API de Riot Games*. Cada registro representa el desempeño técnico individual de un jugador en una partida competitiva específica, proporcionando una granularidad estadística que permite capturar la complejidad táctica de League of Legends. La robustez de este volumen de datos asegura que los modelos de Inteligencia Artificial puedan identificar patrones de victoria significativos, minimizando el riesgo de falsas correlaciones.

Los datos se estructuran en cuatro dimensiones analíticas fundamentales, diseñadas para capturar las diferentes facetas del rendimiento estratégico:

- *Variable Objetivo (`win`)*: Un indicador booleano que define el resultado final de la partida. Es la etiqueta crítica utilizada por los algoritmos de *Aprendizaje Supervisado* para aprender las condiciones de éxito.
- *Métricas Transversales (Normalizadas `per-minute`)*: Incluye variables esenciales como `goldEarned`, `totalMinionsKilled`, `totalDamageDealtToChampions`, `totalDamageTaken`, `damageDealtToEpicMonsters`, `damageDealtToTurrets`, `kills`, `deaths`, `assists` y `visionScore`. La transformación de estos datos en valores por minuto es crucial, ya que compensa la variabilidad intrínseca en la duración de las partidas (`timePlayed`), convirtiendo valores brutos en indicadores de eficiencia relativa.
- *Métricas de Desafío y Equipo (Acumulativas)*: Indicadores avanzados que miden el impacto estratégico individual y la coordinación colectiva, tales como el control de objetivos épicos (`challenge_teamRiftHeraldKills`, `challenge_teamBaronKills`), la participación en la eliminación de campeones enemigos (`challenge_killParticipation`) y la ventaja competitiva obtenida frente al oponente directo en los primeros minutos de juego (`challenge_laningPhaseGoldExpAdvantage`).
- *Contexto de Segmentación Estructural*: La variable `individualPosition` (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY) actúa como el eje de segmentación del dataset. El análisis exploratorio preliminar confirma que los perfiles estadísticos son radicalmente dispares según el rol jugado; un soporte (`UTILITY`) se valida por su puntuación de visión y asistencias, mientras que un carrilero central (`MIDDLE`) o superior (`TOP`) se evalúa prioritariamente por su capacidad de generación de recursos y daño infligido.

Para garantizar una gestión profesional, el proyecto organiza los repositorios físicos de datos en un directorio `data/` independiente del código fuente. Se utiliza el sistema *Data Version Control (DVC)* para asegurar la trazabilidad y persistencia de estos datasets pesados, permitiendo una sincronización exacta entre las versiones del código de modelado y la versión histórica de los datos capturados, cumpliendo con los estándares rigurosos de un entorno de *Data Science*.

== Preparación de los datos

Para garantizar la calidad y robustez de los modelos predictivos integrados en `codigo/modelado_lol.py`, se aplica un pipeline de transformación de datos detallado que asegura la coherencia estadística y la relevancia táctica de la información analizada:

- *Limpieza de Sesgos Competitivos*: Se realiza un filtrado riguroso eliminando registros con jugadores ausentes (`challenge_hadAfkTeammate == 1`) o partidas de duración inferior a los *420 segundos* (7 minutos). Esta depuración es esencial para evitar el "ruido" estadístico en los modelos, ya que las partidas incompletas o desbalanceadas no reflejan patrones de victoria reales basados en el rendimiento técnico, sino en factores externos de abandono.
- *Ingeniería de Atributos (Normalización `per-minute`)*: Dado que la duración de las partidas competitivas oscila ampliamente (desde los 20 hasta los 45 minutos), los valores absolutos de oro o daño no son comparables directamente. El sistema recalcula todas las métricas base dividiéndolas por el tiempo total jugado (`timePlayed` / 60), generando nuevos indicadores de eficiencia denominados *métricas perMin*. Esta normalización permite al sistema entender el ritmo de juego y la densidad de generación de recursos, independientemente de la longitud de la partida, asegurando una base de comparación equitativa.
- *Estandarización de Escala con StandardScaler*: Para asegurar un entrenamiento óptimo de los algoritmos sensibles a la magnitud de los datos —como las Redes Neuronales (MLP), el *K-Nearest Neighbors* (KNN) y el *Support Vector Machine* (SVM)—, se aplica una estandarización de puntuación Z (*Z-score scaling*). Este proceso centra los datos en cero y ajusta la varianza a la unidad, evitando que variables con escalas numéricas grandes (como el daño total) dominen injustamente sobre métricas con rangos menores pero estratégicamente vitales (como el puntaje de visión o kills).
- *División y Muestreo de Datos*: El dataset se divide utilizando una semilla de aleatoriedad fija (`random_state=42`) para garantizar la reproducibilidad absoluta de los hallazgos. Se reserva sistemáticamente un *10 %* de los datos para la fase de prueba (*Test set*), asegurando una validación ciega que certifique la capacidad de generalización de los 9 modelos seleccionados.
- *Arquitectura de Modelos Especializados*: Una decisión de diseño crítica en esta fase es la segmentación de la arquitectura por rol. En lugar de utilizar codificación (*One-Hot Encoding*) para la posición del jugador, el flujo de preprocesamiento divide el dataset según la variable `individualPosition`. Esto permite entrenar y persistir un conjunto de modelos independiente para cada una de las cinco posiciones competitivas (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY), asegurando que cada IA sea calibrada exclusivamente con la lógica táctica y las métricas determinantes de su respectivo rol.

== Modelado

La fase de modelado en `codigo/modelado_lol.py` se fundamenta en un paradigma de *Diversidad Algorítmica*, evaluando simultáneamente 9 arquitecturas distintas para capturar patrones de éxito tanto lineales como altamente complejos. Para cada uno de los cinco roles competitivos de League of Legends, el sistema entrena y persiste un conjunto de modelos que ofrecen diferentes perspectivas analíticas:

- *Algoritmos Estadísticos y Probabilísticos*: Se implementan la *Regresión Logística* (LR) y el *Análisis de Discriminante Lineal* (LDA) para capturar relaciones lineales directas entre el desempeño táctico y la victoria. Complementariamente, se utiliza *Naive Bayes* (NB) para modelar la probabilidad condicional de éxito basándose en la distribución de las variables tácticas.
- *Ensambles y Árboles de Decisión*: Para capturar jerarquías de decisiones críticas y patrones no lineales, se despliega un *Árbol de Decisión* (DT) con profundidad controlada, junto con potentes algoritmos de ensamble como *Random Forest* (RF) y *XGBoost*. Estos últimos son especialmente robustos ante la interacción multidimensional de variables (p. ej., cómo la economía de oro interactúa dinámicamente con el control de objetivos épicos como el Barón Nashor).
- *Instancia y Fronteras de Decisión*: Se utiliza el algoritmo *K-Nearest Neighbors* (KNN), cuya eficacia se optimiza dinámicamente mediante una búsqueda de rejilla (*GridSearchCV*) que selecciona el valor de $k$ (vecinos) más adecuado para la distribución de cada rol específico (evaluando rangos entre 3 y 21). Asimismo, se implementa un *Support Vector Machine* (SVM) basado en `LinearSVC`, el cual se procesa mediante un envoltorio de calibración (`CalibratedClassifierCV`) para asegurar que el modelo no solo clasifique el resultado, sino que devuelva una probabilidad precisa necesaria para el simulador interactivo de la aplicación.
- *Aprendizaje Profundo (Redes Neuronales MLP)*: Se integra una arquitectura de red neuronal utilizando *TensorFlow/Keras*. El Perceptrón Multicapa (MLP) consta de una capa de entrada ajustada al vector de características, dos capas densamente conectadas con funciones de activación *ReLU* (de 32 y 16 neuronas respectivamente) y una capa de salida con activación *Sigmoid* para la clasificación binaria. El entrenamiento se optimiza con el algoritmo *Adam* y una función de pérdida de entropía cruzada binaria (*binary_crossentropy*).

El flujo de modelado está completamente automatizado y desacoplado, permitiendo que cada arquitectura se registre de forma independiente en la infraestructura de seguimiento. Los artefactos resultantes (modelos en formato `.joblib` y `.keras`) se organizan jerárquicamente en el directorio `modelos/{role}/`, asegurando una integridad total entre el entrenamiento y la inferencia en tiempo real.

== Evaluación

La fase de evaluación en `codigo/modelado_lol.py` trasciende la simple medición de precisión para integrar un marco de *Explicabilidad Unificada* y auditoría profesional. Se aplican las siguientes estrategias de validación para asegurar que los modelos sean fidedignos y estratégicamente relevantes para el usuario final:

- *Validación del Desempeño Técnico*: Los 9 modelos de cada rol se someten a pruebas rigurosas utilizando el segmento de datos independizado durante la fase de preparación (*Test set*). Se calcula un perfil completo de métricas de clasificación que incluye *ROC-AUC* (capacidad de discriminación de clases), *Accuracy*, *F1-Score* y precisión. Adicionalmente, el sistema genera de forma automatizada *Matrices de Confusión* (almacenadas en `metricas_resultados/matrices/`) que permiten diagnosticar patrones de error específicos, garantizando la confiabilidad del "diagnóstico del oráculo" antes de su despliegue.
- *Marco de Explicabilidad Agnostica (Permutation Importance)*: Dada la diversidad de arquitecturas involucradas (desde árboles hasta redes neuronales), el sistema implementa la técnica de *Importancia por Permutación de Atributos*. Este método permite calcular el impacto real de cada variable estratégica independientemente del algoritmo utilizado. El proceso técnico consiste en permutar aleatoriamente los valores de una característica específica y medir la degradación resultante en el desempeño del modelo; una mayor caída en la métrica indica que dicha variable es un factor determinante e indispensable para la victoria. Esta unificación es vital para que, en la interfaz de usuario de `app.py`, la visualización de los descriptores de importancia sea consistente y transparente, sin importar la complejidad del modelo que sustente la predicción.
- *Auditoría e Integridad con MLflow*: Cada proceso de evaluación se registra como una ejecución única ("Run") en la infraestructura de seguimiento. Se logean automáticamente todas las métricas finales y los artefactos de resultados, facilitando la auditoría profesional y la comparación histórica entre diferentes iteraciones de entrenamiento. Esta práctica de *MLOps* asegura que el sistema evolucione de forma predecible, permitiendo identificar qué conjuntos de datos y configuraciones algorítmicas ofrecen los diagnósticos más precisos.
- *Jerarquización del Conocimiento Táctico*: El resultado final de esta fase es la identificación de los *factores críticos de victoria* especializados para cada rol. Esta jerarquización permite que el sistema traduzca el procesamiento estadístico en recomendaciones estratégicas claras, indicando al jugador qué áreas tácticas prioritarias operan como palancas de éxito en su nivel competitivo real.

== Despliegue y Resultados

El sistema se despliega mediante una plataforma interactiva centrada en el usuario, incorporando principios de *MLOps* para gestionar el ciclo de vida del proyecto. La implementación técnica en `codigo/app.py` utiliza el *framework* *Streamlit*, optimizado con estilos CSS personalizados para ofrecer una estética futurista y profesional acorde al ecosistema de los eSports de alto rendimiento.

=== Arquitectura de la Interfaz (Frontend)

La interfaz se ha diseñado bajo una estética "Gaming Premium", utilizando tipografías de corte tecnológico (*Orbitron* y *Outfit*) y una paleta de colores neón sobre fondo oscuro que maximiza la legibilidad de métricas complejas. La navegación es modular y reactiva, permitiendo al usuario transitar fluidamente entre el análisis global de la liga y el diagnóstico técnico individualizado de partidas.

// TODO: Captura de pantalla de la interfaz general (Sidebar y Dashboard de Inicio)
#figure(
  image("img/captura_interfaz_general.png", width: 80%),
  caption: [Vista general de la plataforma League Learning con su sistema de navegación modular y estética profesional.],
)

=== Dashboard de Inicio y Rankings

El punto de entrada es un panel de *Rankings Globales* que contextualiza el nivel de juego. Esta sección sistematiza los datos de los mejores jugadores regionales, permitiendo realizar búsquedas de invocadores específicos y visualizar su progresión técnica. El sistema implementa una lógica de ordenamiento jerárquico por rangos competitivos (desde *Emerald* hasta *Challenger*) y una paginación eficiente para manejar grandes volúmenes de datos sin degradar la respuesta visual.

// TODO: Captura de pantalla de la tabla de Rankings con el buscador de invocadores
#figure(
  image("img/captura_rankings.png", width: 80%),
  caption: [Sistema de rankings con búsqueda dinámica y clasificación jerárquica de invocadores.],
)

=== IA & Simulador Multialgoritmo (El Oráculo)

Este módulo constituye el núcleo de inteligencia interactiva del proyecto. A diferencia de las herramientas estadísticas convencionales, el simulador permite a los usuarios interactuar directamente con los 9 modelos de IA entrenados en `modelado_lol.py`. Sus capacidades principales incluyen:

- *Selección Dinámica de Algoritmo*: Capacidad de alternar entre diferentes arquitecturas de IA (p. ej., XGBoost vs. Redes Neuronales) para validar predicciones.
- *Visualización de Importancia Variable*: Gráficos interactivos de *Plotly* que muestran en tiempo real qué factores determinan el diagnóstico actual.
- *Diagnóstico el Oráculo*: El usuario puede ajustar manualmente indicadores tácticos (oro por minuto, visión o daño) y observar cómo varía la probabilidad de victoria a través de un *gráfico tipo Gauge* altamente visual.

// TODO: Captura de pantalla del módulo IA & Simulador, mostrando el gráfico de importancia de variables y el Gauge de probabilidad
#figure(
  image("img/captura_simulador.png", width: 80%),
  caption: [Simulador dinámico con análisis de sensibilidad táctica y diagnóstico de probabilidad de victoria.],
)

=== Diagnóstico Individual y Perfil de Jugador

La plataforma permite una inmersión profunda en el historial táctico de cualquier jugador. El sistema procesa las partidas recientes y genera etiquetas automáticas de inteligencia basadas en el rendimiento (como *⭐ MVP*, *💥 Carry* o *👁️ Visión*), fundamentadas en los umbrales de los modelos de IA. Esta sección traduce el histórico de partidas en diagnósticos específicos de probabilidad de éxito, facilitando al jugador identificar qué variables críticas operan como "palancas de victoria" en su nivel de juego actual.

// TODO: Captura de pantalla del perfil de un jugador, mostrando su historial de partidas analizado por la IA
#figure(
  stack(dir: ttb, spacing: 1.5mm, image("img/perfil1.png", width: 72%), image("img/perfil2.png", width: 72%), image(
    "img/perfil3.png",
    width: 72%,
  )),
  caption: [Perfil individualizado con diagnóstico de partidas recientes y etiquetas de desempeño automatizadas por la IA.],
)

=== Integración MLOps y Gestión de Versiones (MLflow)

Una innovación crítica del despliegue es el selector de versiones de Inteligencia Artificial que consulta en tiempo real los experimentos registrados en *MLflow*. Esto permite una trazabilidad total, habilitando al usuario para "viajar en el tiempo" entre diferentes versiones de entrenamiento y datasets. La infraestructura descarga automáticamente los artefactos (modelos, escaladores y métricas) desde el servidor de seguimiento, asegurando que el despliegue sea siempre profesionalmente coherente con la última iteración validada del sistema.

// TODO: Captura de pantalla del selector de versiones (MLOps) en el sidebar, mostrando diferentes "Runs" de MLflow
#figure(
  image("img/captura_mlops.png", width: 70%),
  caption: [Sistema de gestión de versiones de IA sincronizado con MLflow para trazabilidad absoluta de los modelos.],
)

= Herramientas y técnicas

El éxito técnico de *League Learning* se fundamenta en la integración de un ecosistema de herramientas de vanguardia, seleccionadas por su capacidad para manejar grandes volúmenes de datos y asegurar la escalabilidad del sistema predictivo. A continuación, se detallan los componentes del *stack* tecnológico utilizado:

- *Ecosistema de Lenguaje y Datos*:
  - *Python 3.10+*: Seleccionado como el lenguaje núcleo debido a su madurez y versatilidad en el manejo de estructuras de datos y su soporte nativo para la mayoría de las librerías de IA actuales.
  - *Pandas y NumPy*: Utilizados para la manipulación matricial de los registros y el preprocesamiento de los 42,000 datos de la API, permitiendo operaciones vectorizadas de alta velocidad para la normalización *per-minute*.

- *Inteligencia Artificial y Modelado*:
  - *Scikit-Learn*: Provee la base para la mayoría de los algoritmos de clasificación (LR, RF, SVM, KNN, etc.), así como las utilidades fundamentales para el escalamiento de datos y la validación cruzada.
  - *XGBoost*: Un algoritmo de *Extreme Gradient Boosting* de alto rendimiento, crítico para capturar patrones técnicos no lineales en el dataset de eSports con alta precisión.
  - *TensorFlow / Keras*: La infraestructura utilizada para el diseño, compilación y entrenamiento del Perceptrón Multicapa (Red Neuronal MLP), optimizada para el aprendizaje profundo.

- *Infraestructura MLOps y Gestión*:
  - *MLflow*: Herramienta central para el rastreo de experimentos. Permite loguear métricas, parámetros y registrar versiones de modelos, garantizando un ciclo de vida de desarrollo profesional, reproducible y auditable.
  - *Data Version Control (DVC)*: Implementado para la gestión de versiones de los archivos de datos de gran tamaño, permitiendo una sincronización eficiente del dataset sin saturar el sistema de control de versiones de código.

- *Interfaz de Usuario y Visualización*:
  - *Streamlit*: *Framework* principal para el despliegue de la aplicación interactiva. Su naturaleza reactiva permite integrar modelos de Python directamente en una interfaz web dinámica y moderna.
  - *Plotly*: Utilizado para la generación de gráficos interactivos, esenciales para la visualización de la importancia de variables y el diagnóstico en tiempo real mediante el simulador.

- *Fuentes de Datos*:
  - *API de Riot Games*: Fuente primaria de datos estadísticos, proporcionando acceso a la granularidad necesaria para el entrenamiento de modelos profesionales.

= Conclusiones

El desarrollo del proyecto *League Learning* ha demostrado que la integración de técnicas avanzadas de Inteligencia Artificial y metodologías de ingeniería de datos permite transformar la complejidad táctica de League of Legends en diagnósticos estratégicos accionables y de alta fidelidad. A través de este trabajo, se han consolidado las siguientes conclusiones fundamentales:

1. *Eficacia del Enfoque Multialgorítmico*: La evaluación simultánea de 9 arquitecturas de Machine Learning ha validado que no existe un "algoritmo universal" para la diversidad táctica del juego. Mientras que los modelos lineales capturan eficazmente la base de la economía, las estructuras complejas (como XGBoost y Redes Neuronales MLP) son indispensables para entender la dinámica no lineal de los roles de alto impacto. La arquitectura segmentada por posición competitiva garantiza que cada diagnóstico sea fiel a la lógica interna de cada rol.

2. *Democratización del Análisis de Alto Rendimiento*: El simulador prospectivo y la plataforma interactiva eliminan la barrera de entrada al análisis estratégico profesional. Lo que antes requería un equipo de analistas técnicos dedicados, ahora es accesible mediante una interfaz reactiva que permite a cualquier jugador identificar sus "palancas de victoria" personales basadas en evidencia matemática rigurosa y no en percepciones subjetivas.

3. *Explicabilidad como Pilar de Confianza*: El uso de *Permutation Importance* ha sido el factor determinante para el éxito del proyecto. Al resolver técnicamente el problema de la "caja negra" de la IA, el sistema no solo predice un resultado probabilístico, sino que educa al usuario sobre la importancia relativa de variables críticas (visión, objetivos, ventaja de líneas), fomentando una cultura de entrenamiento estratégico fundamentado en datos accionables.

4. *Sostenibilidad y Rigor Operativo*: La implementación de un flujo de trabajo de *MLOps* mediante *MLflow* y *DVC* asegura que *League Learning* sea un sistema resiliente y escalable. La trazabilidad total de cada experimento y la capacidad de gestionar versiones de la "Inteligencia de la IA" garantizan que la plataforma pueda adaptarse rápidamente a los constantes parches de balance de Riot Games, manteniendo la vigencia del diagnóstico táctico conforme evoluciona el videojuego.

En conclusión, este proyecto trasciende el análisis estadístico convencional para constituirse como una herramienta funcional de inteligencia competitiva. La transición definitiva de la intuición hacia la toma de decisiones fundamentada en la ciencia de datos representa la evolución necesaria en el entrenamiento de los eSports modernos, posicionando a *League Learning* como una plataforma capaz de transformar el potencial bruto en rendimiento competitivo optimizado.

= Anexos

== Repositorio y Transparencia Técnica

La implementación completa, incluyendo los scripts de entrenamiento multialgoritmo, la lógica del simulador, el pipeline de MLOps, la interfaz de usuario y las instrucciones de uso, se encuentra disponible para auditoría y contribución en:

- *Repositorio de GitHub*: #link("https://github.com/Zinko5/league-learning")[github.com/Zinko5/league-learning]
- *Diccionario Técnico*: El documento `diccionario.md` detalla la semántica y el origen de las 35+ variables procesadas desde la API.

== Guía de Operación y Escalamiento

La plataforma está diseñada para ser desplegada de forma sencilla en entornos locales para análisis táctico:

1. *Entrenamiento*: Ejecutar `python3 codigo/modelado_lol.py` para recalibrar los modelos por rol e iniciar el registro de experimentos en el servidor local de *MLflow*.
2. *Visualización*: Ejecutar `streamlit run codigo/app.py` para lanzar la interfaz del "Oráculo" y acceder al simulador de probabilidad dinámica.

== Cumplimiento y Aviso Legal (Riot Games)

Debido al uso de recursos oficiales de Riot Games (API de Riot Games para obtener datos de partidas), se establece la siguiente declaración de conformidad regulatoria de acuerdo a la política de desarrolladores de Riot Games:

#link("https://developer.riotgames.com/policies/general")[developer.riotgames.com/policies/general]

#block(
  fill: luma(245),
  inset: 12pt,
  radius: 4pt,
  stroke: luma(200),
  text(style: "italic", size: 0.85em)[
    *League Learning* isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.
  ],
)
