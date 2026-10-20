# Revisión enfocada y pruebas — Pre-entrega M9

**Repositorio:** <https://github.com/denparra/ds.pre-entrega-9>

## Criterios de la consigna

| Criterio | Implementación | Verificación |
|---|---|---|
| README con descripción del dataset | `README.md` documenta Titanic, `Survived`, las columnas usadas, exclusiones y fuente. | Revisado. |
| `Pipeline` con transformaciones | El pipeline numérico imputa y escala; el categórico imputa y codifica. Ambos quedan dentro del `ColumnTransformer` y del pipeline final. | `test_pipeline_diferencia_columnas_y_transformaciones` pasa. |
| Tratamiento numérico y categórico | `ColumnTransformer` enruta las siete columnas a ramas separadas. | Prueba de estructura del pipeline. |
| Split antes de ajustar el preprocesamiento | `train_test_split` ocurre antes de `pipeline.fit(X_train, y_train)`. El test comprueba que la mediana aprendida por el imputador corresponde a `X_train`. | `test_entrena_con_train_evalua_test_y_persiste_pipeline` pasa. |
| Predicción de datos nuevos | `src.predict` carga el pipeline y ejecuta `.predict()`; las categorías nuevas se ignoran de forma segura. | Prueba con `Embarked="X"` pasa. |
| Persistencia y métricas | El pipeline completo se guarda con `joblib`; se informan accuracy, métricas por clase y matriz de confusión. | Prueba de guardado/carga y ejecución de entrenamiento. |

## Ejecuciones locales

Entorno usado: Python 3.13.1, pandas 3.0.6, scikit-learn 1.9.1 y joblib 1.6.0.

```text
python -m unittest discover -s tests -v
Ran 4 tests — OK
```

La ejecución de `python -m src.train_pipeline` produjo:

- Filas de entrenamiento: **712**.
- Filas de prueba: **179**.
- Accuracy: **0.804**.
- Matriz de confusión (filas reales, columnas predichas): `[[98, 12], [23, 46]]`.
- La ejecución de `python -m src.predict` cargó el modelo y predijo correctamente con una categoría no vista (`Embarked="X"`), sin error de formato.

El artefacto `models/titanic_pipeline.joblib` se genera al entrenar y está excluido de Git; se puede reconstruir localmente a partir del código y del CSV incluido.

## Fuente de datos

`data/titanic.csv` proviene del [dataset Titanic publicado por DataScienceDojo](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv). La fuente está citada también en el README del proyecto.
