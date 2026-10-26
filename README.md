**Actividad:** 6
**Alumno:** Manuel de Jesus Ramiro Velázquez Tapia  
**Matrícula:** AL07183522  
**Docente:** Luis Ariel Vazquez Piña  

---
Este repositorio contiene la solución completa de experimentación, evaluación rigurosa y validación estadística para la **Actividad 6** de la materia *Gestión de Proyectos de Inteligencia Artificial*. El proyecto aplica técnicas de Machine Learning supervisado en un contexto financiero crítico: la **detección y mitigación oportuna de transacciones fraudulentas en banca digital**.

---

## 1. Descripción del Proyecto y Arquitectura

* **Contexto de Negocio:** Detección de fraude transaccional con costos asimétricos severos:
  * **Falso Negativo (FN):** Pérdida patrimonial directa estimada en **$1,500 USD** por evento.
  * **Falso Positivo (FP):** Fricción operativa y costo de soporte telefónico estimado en **$25 USD** por alerta.
  * **Ratio de Penalización:** $1\text{ FN} \equiv 60\text{ FP}$.
* **Objetivo SMART:** Lograr en validación cruzada estratificada un $\text{Recall} \ge 85\%$, $\text{F1-Score} \ge 0.65$ y $\text{AUC-ROC} \ge 0.90$, reduciendo las pérdidas por fraude no detectado en más de un $40\%$ frente al baseline en un ciclo de 4 semanas.

---

## 2. Entorno de Ejecución y Librerías Utilizadas

El pipeline experimental está desarrollado en **Python 3.12** utilizando el ecosistema científico estándar de código abierto.

### Tabla de Dependencias y Versiones

| Librería / Módulo | Propósito y Uso en el Proyecto |
| :--- | :--- |
| **`python`** ($\ge 3.10$, probado en `3.12`) | Intérprete base de ejecución del código y notebooks. |
| **`scikit-learn`** | Algoritmos de clasificación (`RandomForestClassifier`, `LogisticRegression`, `DummyClassifier`), partición estratificada (`train_test_split`, `StratifiedKFold`), cálculo de métricas (`confusion_matrix`, `precision_score`, `recall_score`, `f1_score`, `roc_curve`, `auc`, `roc_auc_score`). |
| **`numpy`** | Manejo de matrices numéricas, operaciones vectorizadas de costos y manipulación de umbrales. |
| **`pandas`** | Estructuración del dataset transaccional, generación de tablas de sensibilidad y registro tipo MLflow. |
| **`scipy`** | Cálculo de intervalos de confianza al 95% para métricas de validación cruzada y funciones de distribución normal acumulada. |
| **`statsmodels`** | Implementación del contraste de hipótesis para pruebas A/B mediante prueba $Z$ de dos proporciones independientes (`proportions_ztest`). |
| **`matplotlib`** | Generación de visualizaciones estáticas: matrices de confusión comparativas, curvas ROC con coordenadas y gráficas de A/B testing. |
| **`seaborn`** | Mapas de calor para matrices de confusión e histogramas de distribuciones de probabilidad predicha. |

### Instalación de Requerimientos

Para replicar el entorno localmente:
```bash
pip install numpy pandas scikit-learn scipy statsmodels matplotlib seaborn jupyter
```

---

## 3. Modelos Evaluados y Configuraciones de Hiperparámetros

Se implementó un diseño experimental comparativo progresivo (baselines heurísticos y lineales vs. ensamble no lineal):

### 1. Modelo Baseline Heurístico (Nulo)
* **Clase:** `sklearn.dummy.DummyClassifier`
* **Estrategia:** `strategy='most_frequent'`
* **Configuración:** Predice invariablemente la clase mayoritaria (clase $0$, transacciones legítimas).
* **Propósito:** Demostrar la *Paradoja de la Exactitud* (*Accuracy Paradox*) en datos desbalanceados.

### 2. Modelo Baseline Lineal
* **Clase:** `sklearn.linear_model.LogisticRegression`
* **Hiperparámetros:**
  * `class_weight='balanced'`: Ponderación inversa a las frecuencias de clase para compensar el desbalance.
  * `max_iter=1000`: Garantiza convergencia del gradiente.
  * `solver='lbfgs'`: Optimizador cuasi-Newton estándar para regresión logística.
  * `random_state=42`: Control estricto de reproducibilidad estocástica.
* **Propósito:** Proveer un punto de referencia estadístico interpretable y lineal.

### 3. Modelo Propuesto (Ensamble No Lineal Ganador)
* **Clase:** `sklearn.ensemble.RandomForestClassifier`
* **Hiperparámetros y Justificación:**
  * `n_estimators=150`: Número suficiente de árboles para estabilizar la varianza del ensamble sin incurrir en latencia excesiva.
  * `max_depth=7`: Profundidad máxima restringida para evitar sobreajuste (*overfitting*) en datos tabulares de 12 variables.
  * `min_samples_split=10`: Requiere al menos 10 muestras para generar una nueva división interna.
  * `min_samples_leaf=4`: Regularización que previene hojas con ruido o casos atípicos aislados.
  * `class_weight='balanced'`: Ajuste automático de pesos proporcional a $w_j = \frac{N}{2 \cdot n_j}$.
  * `random_state=42`: Semilla fija para reproducibilidad total.
  * `n_jobs=-1`: Paralelización multinúcleo para entrenamiento acelerado.

---

## 4. Configuraciones Experimentales y Metodología

### 4.1. Generación y Partición de Datos
* **Tamaño Muestral:** 2,500 transacciones sintéticas con 12 variables explicativas multivariadas (monto, velocidad 24h, desvío de ticket, distancia IP, score de dispositivo nuevo, antigüedad de cuenta, hora nocturna de riesgo, tipo de comercio internacional, reintentos).
* **Distribución de Clases:**
  * Legítimas (Clase 0): $87.6\%$ ($2,190$ observaciones).
  * Fraude (Clase 1): $12.4\%$ ($310$ observaciones).
* **Estrategia de Partición:** División estratificada 70% entrenamiento ($1,750$ registros) y 30% prueba ($750$ registros) preservando idéntica proporción de fraudes ($12.4\%$, 93 eventos en test).

### 4.2. Función de Costos y Calibración de Umbrales de Decisión
En lugar de aceptar ciegamente el umbral estándar de probabilidad ($t = 0.50$), se evaluó una rejilla de umbrales en el rango $t \in [0.10, 0.80]$ mediante la función de costo financiero:
$$\text{Costo Total} = (\text{FN} \times \$1,500) + (\text{FP} \times \$25)$$

* **Umbral Estándar ($t = 0.50$):** Recall $65.59\%$, $32$ Falsos Negativos (pérdida de $\$48,000$ USD), Costo total: $\$48,600$ USD.
* **Umbral Operativo Recomendado ($t = 0.35$):** Recall $83.87\%$, $15$ Falsos Negativos (pérdida de $\$22,500$ USD), $69$ Falsos Positivos, Costo total: $\$24,225$ USD (**reducción del $50.2\%$ en costos totales**).
* **Umbral de Mínimo Costo Absoluto ($t = 0.20$):** Recall $93.55\%$, solo $6$ Falsos Negativos, Costo total: $\$13,725$ USD (mayor volumen de alertas preventivas).

### 4.3. Validación Cruzada Estratificada ($k = 5$)
* **Configuración:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
* **Métricas Evaluadas por Pliegue:** Recall, Precision, F1-Score y AUC-ROC a umbral óptimo calibrado.
* **Inferencia Estadística:** Cálculo de media muestral ($\mu$), desviación estándar ($\sigma$) e Intervalo de Confianza al 95%:
  $$\text{IC}_{95\%} = \bar{x} \pm 1.96 \left(\frac{s}{\sqrt{k}}\right)$$

### 4.4. Diseño de Pruebas A/B en Producción
* **Población del Experimento:** 10,000 transacciones bancarias en vivo durante 14 días.
* **Asignación Aleatoria 50/50:**
  * **Grupo Control (A):** Modelo previo (Regresión Logística, $t = 0.50$, $5,000$ transacciones).
  * **Grupo Tratamiento (B):** Modelo propuesto (Random Forest, $t = 0.35$, $5,000$ transacciones).
* **Hipótesis Estadística:**
  * $H_0: p_B - p_A \le 0$ (El modelo propuesto no incrementa la tasa de detección).
  * $H_1: p_B - p_A > 0$ (El modelo propuesto incrementa significativamente la tasa de detección).
* **Prueba Aplicada:** Prueba $Z$ de dos proporciones independientes con nivel de significancia $\alpha = 0.05$.

---

## 5. Resumen de Resultados Experimentales

### 5.1. Comparativa Offline (Conjunto de Prueba $N=750$)
* **Dummy Baseline:** Accuracy = $87.60\%$, Recall = $0.00\%$, F1 = $0.0000$, AUC = $0.5000$.
* **Regresión Logística:** Accuracy = $71.73\%$, Recall = $70.97\%$, F1 = $0.3837$, AUC = $0.7749$.
* **Random Forest ($t=0.50$):** Accuracy = $92.53\%$, Recall = $65.59\%$, F1 = $0.6854$, AUC = $0.9200$.
* **Random Forest ($t=0.35$ Opt):** Accuracy = **$88.80\%$**, Recall = **$83.87\%$**, F1 = **$0.6500$**, AUC = **$0.9200$**.

### 5.2. Validación Cruzada (5 Pliegues)
* **Recall Promedio:** $81.66\% \pm 9.64\%$ | IC 95%: $[73.21\%, 90.11\%]$.
* **F1-Score Promedio:** $0.6673 \pm 0.0583$ | IC 95%: $[0.6162, 0.7184]$.
* **AUC-ROC Promedio:** $0.9289 \pm 0.0237$ | IC 95%: $[0.9081, 0.9497]$.

### 5.3. Inferencia de Prueba A/B Online
* **Tasa de Detección Control A:** $73.42\%$ ($384/523$ fraudes interceptados).
* **Tasa de Detección Tratamiento B:** $92.69\%$ ($571/616$ fraudes interceptados).
* **Incremento Neto en Detección:** $+19.27\%$.
* **Estadístico $Z$:** $+8.8069$ | **$p\text{-valor}$:** $6.43 \times 10^{-19}$ ($p < 0.001$).
* **Ahorro Financiero Directo:** **$144,775 USD** en el periodo de 14 días (proyección anual $> \$3.76\text{M USD}$).

---

## 6. Estructura de Archivos del Proyecto

```
.
├── README.md                    # Documentación técnica del proyecto
├── CHANGELOG.md                 # Registro de cambios y reporte técnico
├── datos/
│   └── datos_fraude.csv         # Datos de transacciones para el análisis de fraude
├── fuentes/
│   └── GP_Actividad6.ipynb      # Notebook con el análisis, código y visualizaciones
└── imagenes/
    └── tablero_trello.pdf       # Evidencia del tablero de seguimiento del proyecto
```

---

## 7. Instrucciones para Ejecutar los Experimentos

### Ejecución del Script Python
Para reproducir de principio a fin los datos sintéticos, entrenamiento, cálculo de matrices, optimización de umbrales, validación cruzada y simulación A/B:
```bash
python experimentacion_actividad6.py
```

### Ejecución del Jupyter Notebook
Para interactuar visualmente con las celdas, gráficos y tablas paso a paso:
```bash
jupyter notebook experimentacion_actividad6.ipynb
```
O bien abrir el archivo en **Google Colab**, **VS Code** o **JupyterLab** y seleccionar `Run All Cells`.
