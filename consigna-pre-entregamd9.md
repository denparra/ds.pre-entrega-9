# Actividad: Pre-entrega — Construcción de un Pipeline de Preprocesamiento Robusto

## Objetivo

Consolidar las habilidades de automatización de limpieza y transformación de datos mediante **Scikit-Learn Pipelines**. En lugar de realizar transformaciones manuales y dispersas, se construirá una arquitectura reutilizable que prepare los datos para un modelo de Machine Learning.

El propósito es evitar la fuga de datos (*data leakage*) y asegurar que el preprocesamiento sea consistente tanto durante el entrenamiento como durante la inferencia, encapsulando la lógica en un único objeto `Pipeline`.

## Entregable

Un **repositorio público en GitHub** que contenga:

- Un Jupyter Notebook o script de Python con la implementación del pipeline.
- Un archivo `README.md` explicativo, con la descripción del dataset utilizado.
- El código necesario para exportar el pipeline entrenado mediante `joblib`.

## Pasos sugeridos

1. **Carga y exploración:** utilizar un conjunto de datos —por ejemplo, Housing, Iris o uno propio persistido en módulos anteriores— e identificar las columnas numéricas y categóricas.
2. **Definición de transformadores:** crear un `ColumnTransformer` que realice un tratamiento diferenciado:
   - **Variables numéricas:** imputación de valores nulos con la media o mediana y escalado con `StandardScaler` o `MinMaxScaler`.
   - **Variables categóricas:** imputación con el valor más frecuente y codificación con `OneHotEncoder`.
3. **Ensamblaje del pipeline:** integrar el preprocesador en un `Pipeline` final que incluya al menos un estimador básico, como Regresión Lineal o Logística.
4. **Validación:** ejecutar `train_test_split`, entrenar el pipeline completo y generar un reporte básico de métricas.
5. **Persistencia:** incluir el código que exporta el pipeline entrenado con `joblib`.

## Criterios de revisión

- [ ] El repositorio contiene un `README.md` con la descripción del dataset utilizado.
- [ ] El código utiliza `Pipeline` de Scikit-Learn para encadenar al menos dos pasos de transformación.
- [ ] Se implementa un `ColumnTransformer` que diferencia explícitamente el tratamiento de variables numéricas y categóricas.
- [ ] Los datos se dividen en entrenamiento y prueba antes de ajustar cualquier parámetro de preprocesamiento.
- [ ] El pipeline final permite ejecutar `.predict()` sobre datos nuevos sin errores de formato.

## Errores comunes

- Aplicar `fit_transform` al dataset completo antes de dividirlo; esto causa fuga de datos.
- No manejar categorías nuevas o desconocidas en `OneHotEncoder`; utilizar `handle_unknown="ignore"`.

## Entrega final

Subir el código a un repositorio público siguiendo una estructura limpia de archivos `.py` o `.ipynb`.