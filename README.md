# Pre-entrega M9 — Pipeline robusto de preprocesamiento

Este proyecto entrena un clasificador para estimar si un pasajero del Titanic sobrevivió. El preprocesamiento se integra al modelo mediante Scikit-Learn Pipelines para evitar fuga de datos y repetir las mismas transformaciones al evaluar y predecir.

**Repositorio público:** <https://github.com/denparra/ds.pre-entrega-9>

## Inicio rápido

Desde esta carpeta, en Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train_pipeline
.\.venv\Scripts\python.exe -m src.predict
```

En macOS o Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.train_pipeline
.venv/bin/python -m src.predict
```

El entrenamiento muestra las métricas y guarda el pipeline completo en `models/titanic_pipeline.joblib`. Ese artefacto se genera localmente y se excluye de Git; puede reconstruirse ejecutando el script de entrenamiento.

## Dataset

- **Archivo:** `data/titanic.csv` (891 pasajeros).
- **Objetivo:** `Survived`, donde `0` indica que no sobrevivió y `1` que sobrevivió.
- **Fuente:** [DataScienceDojo — Titanic dataset](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv), copia pública del conjunto de entrenamiento clásico de Titanic.
- **Variables utilizadas:** `Age`, `SibSp`, `Parch`, `Fare`, `Pclass`, `Sex` y `Embarked`.
- **Variables excluidas:** `PassengerId` es un identificador; `Name` y `Ticket` tienen alta cardinalidad; `Cabin` tiene muchos valores faltantes. Se omiten para mantener el ejercicio enfocado en el preprocesamiento.

`Pclass` se trata como categórica aunque sus valores sean números: representa clases de pasaje, no una magnitud continua.

## Preprocesamiento

| Tipo de variable | Columnas | Pasos dentro del pipeline |
|---|---|---|
| Numéricas | `Age`, `SibSp`, `Parch`, `Fare` | Imputar faltantes con la mediana → `StandardScaler` |
| Categóricas | `Pclass`, `Sex`, `Embarked` | Imputar faltantes con el valor más frecuente → `OneHotEncoder(handle_unknown="ignore")` |

Un `ColumnTransformer` aplica ambas ramas y el `Pipeline` las conecta con un clasificador de Regresión Logística. Los imputadores y el escalador aprenden sus parámetros únicamente al ajustar el pipeline con el conjunto de entrenamiento.

## División y evaluación

El script separa primero los datos con `train_test_split` y mantiene la proporción de clases usando `stratify`. Luego ajusta el pipeline solo con `X_train` e `y_train`, y evalúa las predicciones sobre `X_test`. Informa:

- Exactitud (*accuracy*).
- Precision, recall y F1 por clase.
- Matriz de confusión.

La semilla fija (`random_state=42`) permite reproducir la partición. Las métricas son una referencia didáctica, no una garantía de desempeño para decisiones reales.

## Predicción con datos nuevos

`src.predict` carga el pipeline con `joblib` y ejecuta `.predict()` sobre un pasajero de ejemplo. El ejemplo contiene una categoría de `Embarked` no presente en el entrenamiento; `handle_unknown="ignore"` permite procesarla sin error.

Al llamar a `predecir()` desde Python, cada registro debe incluir las siete columnas descritas arriba. No se imputan ni escalan los valores manualmente: el pipeline guardado realiza esas transformaciones.

## Pruebas

```powershell
py -m unittest discover -s tests -v
```

Las pruebas verifican las ramas numérica y categórica, el entrenamiento y la persistencia con `joblib`, la predicción con categorías nuevas y la validación de columnas requeridas.

## Estructura

```text
pre-entregamd9/
├── consigna-pre-entregamd9.md
├── data/titanic.csv
├── src/
│   ├── __init__.py
│   ├── train_pipeline.py
│   └── predict.py
├── tests/test_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

La revisión de los criterios de la consigna y los resultados de las pruebas están en [`INFORME_REVISION_Y_PRUEBAS.md`](./INFORME_REVISION_Y_PRUEBAS.md).
