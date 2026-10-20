"""Entrena, evalúa y guarda un pipeline de clasificación para Titanic."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "titanic.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "titanic_pipeline.joblib"

TARGET_COLUMN = "Survived"
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def construir_pipeline() -> Pipeline:
    """Crea las transformaciones por tipo de columna y el clasificador."""
    pipeline_numerico = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
        ]
    )
    pipeline_categorico = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="most_frequent")),
            ("codificador", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("numericas", pipeline_numerico, NUMERIC_FEATURES),
            ("categoricas", pipeline_categorico, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("clasificador", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def entrenar(
    ruta_datos: Path = DEFAULT_DATA_PATH,
    ruta_modelo: Path = DEFAULT_MODEL_PATH,
) -> dict[str, object]:
    """Divide, entrena solo con train, evalúa con test y guarda el pipeline."""
    datos = pd.read_csv(ruta_datos)
    columnas_requeridas = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    faltantes = columnas_requeridas - set(datos.columns)
    if faltantes:
        raise ValueError(
            "Al dataset le faltan columnas requeridas: " + ", ".join(sorted(faltantes))
        )

    X = datos[FEATURE_COLUMNS].copy()
    y = datos[TARGET_COLUMN].copy()
    if y.isna().any():
        raise ValueError(f"La columna objetivo {TARGET_COLUMN} tiene valores faltantes.")

    # Primero se separan los datos; el preprocesamiento aprende solo de train.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = construir_pipeline()
    pipeline.fit(X_train, y_train)
    predicciones = pipeline.predict(X_test)

    ruta_modelo = Path(ruta_modelo)
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ruta_modelo)

    return {
        "accuracy": float(accuracy_score(y_test, predicciones)),
        "classification_report": classification_report(
            y_test,
            predicciones,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predicciones).tolist(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "model_path": ruta_modelo,
    }


def main() -> None:
    resultados = entrenar()
    print(f"Filas de entrenamiento: {resultados['train_rows']}")
    print(f"Filas de prueba: {resultados['test_rows']}")
    print(f"Accuracy: {resultados['accuracy']:.3f}")
    print("\nReporte de clasificación:")
    print(resultados["classification_report"])
    print("Matriz de confusión [filas=real, columnas=predicho]:")
    print(resultados["confusion_matrix"])
    print(f"\nPipeline guardado en: {resultados['model_path']}")


if __name__ == "__main__":
    main()
