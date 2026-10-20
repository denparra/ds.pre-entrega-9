"""Carga el pipeline guardado y predice sobre pasajeros nuevos."""

from pathlib import Path

import joblib
import pandas as pd

from src.train_pipeline import DEFAULT_MODEL_PATH, FEATURE_COLUMNS


def predecir(
    registros: list[dict[str, object]],
    ruta_modelo: Path = DEFAULT_MODEL_PATH,
) -> list[int]:
    """Predice 0/1 sin preprocesar manualmente las columnas de entrada."""
    ruta_modelo = Path(ruta_modelo)
    if not ruta_modelo.is_file():
        raise FileNotFoundError(
            f"No existe el modelo {ruta_modelo}. Ejecuta primero: py -m src.train_pipeline"
        )

    pipeline = joblib.load(ruta_modelo)
    nuevos = pd.DataFrame(registros)
    faltantes = set(FEATURE_COLUMNS) - set(nuevos.columns)
    if faltantes:
        raise ValueError(
            "A los registros nuevos les faltan columnas: " + ", ".join(sorted(faltantes))
        )

    predicciones = pipeline.predict(nuevos[FEATURE_COLUMNS])
    return [int(prediccion) for prediccion in predicciones]


def main() -> None:
    pasajero_nuevo = {
        "Age": 29,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 25.0,
        "Pclass": 3,
        "Sex": "female",
        # Valor intencionalmente no visto durante el entrenamiento.
        "Embarked": "X",
    }
    resultado = predecir([pasajero_nuevo])[0]
    etiqueta = "sobrevivió" if resultado == 1 else "no sobrevivió"
    print(f"Predicción para el pasajero de ejemplo: {resultado} ({etiqueta}).")


if __name__ == "__main__":
    main()
