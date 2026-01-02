# Análisis Predictivo de Riesgo de Fuga (Customer Churn)

Este proyecto desarrolla un modelo de aprendizaje automático diseñado para identificar patrones de deserción en una cartera de más de 3,000 registros de clientes. El objetivo principal es transformar datos históricos en información estratégica para optimizar la toma de decisiones y las políticas de retención de la organización.

---

## Tecnologías Utilizadas

* **Python**: Lenguaje principal para el procesamiento de datos y el modelado predictivo.
* **Pandas y NumPy**: Herramientas utilizadas para la limpieza de datos, manipulación de matrices y preprocesamiento de variables.
* **Scikit-Learn**: Implementación y validación del modelo de Regresión Logística para clasificación binaria.
* **Matplotlib**: Generación de visualizaciones para el Análisis Exploratorio de Datos (EDA) y la identificación de tendencias.

---

## Resultados y Métricas de Desempeño

El modelo fue validado mediante métricas de clasificación estándar, demostrando una alta confiabilidad en la identificación del estado de los clientes:

* **Exactitud (Accuracy)**: 86.6%.
* **Métricas Adicionales**: Evaluación basada en Precision, Recall y F1-Score para asegurar la capacidad de detección de casos críticos.
* **Hallazgos Clave**: El análisis estadístico permitió identificar que la falta de renovación de contrato y los cargos mensuales elevados son las variables con mayor impacto en el riesgo de fuga.

---

## Estructura del Repositorio

* **analisis_riesgo.py**: Código fuente que contiene la lógica de carga, tratamiento de datos y entrenamiento del modelo.
* **customer_churn.csv**: Conjunto de datos utilizado para el entrenamiento y prueba del sistema.
