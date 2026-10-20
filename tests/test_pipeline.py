"""Pruebas del preprocesamiento, la persistencia y la inferencia."""

import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.predict import predecir
from src.train_pipeline import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    construir_pipeline,
    entrenar,
)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.model_path = Path(self.temp_dir.name) / "pipeline.joblib"

    def test_pipeline_diferencia_columnas_y_transformaciones(self) -> None:
        pipeline = construir_pipeline()
        self.assertIsInstance(pipeline, Pipeline)
        preprocesador = pipeline.named_steps["preprocesador"]
        transformadores = {
            nombre: (transformer, columnas)
            for nombre, transformer, columnas in preprocesador.transformers
        }

        self.assertEqual(transformadores["numericas"][1], NUMERIC_FEATURES)
        self.assertEqual(transformadores["categoricas"][1], CATEGORICAL_FEATURES)
        self.assertEqual(
            [nombre for nombre, _ in transformadores["numericas"][0].steps],
            ["imputador", "escalador"],
        )
        self.assertEqual(
            [nombre for nombre, _ in transformadores["categoricas"][0].steps],
            ["imputador", "codificador"],
        )
        self.assertEqual(
            transformadores["categoricas"][0]
            .named_steps["codificador"]
            .handle_unknown,
            "ignore",
        )

    def test_entrena_con_train_evalua_test_y_persiste_pipeline(self) -> None:
        resultados = entrenar(DEFAULT_DATA_PATH, self.model_path)

        self.assertTrue(self.model_path.is_file())
        self.assertGreater(resultados["train_rows"], resultados["test_rows"])
        self.assertGreater(resultados["test_rows"], 0)
        self.assertGreaterEqual(resultados["accuracy"], 0.0)
        self.assertLessEqual(resultados["accuracy"], 1.0)
        self.assertEqual(len(resultados["confusion_matrix"]), 2)
        self.assertIsInstance(joblib.load(self.model_path), Pipeline)

        # Comprueba que el imputador numérico aprendió del subconjunto train.
        datos = pd.read_csv(DEFAULT_DATA_PATH)
        X_train, _, _, _ = train_test_split(
            datos[FEATURE_COLUMNS],
            datos[TARGET_COLUMN],
            test_size=0.2,
            random_state=42,
            stratify=datos[TARGET_COLUMN],
        )
        pipeline = joblib.load(self.model_path)
        imputador = (
            pipeline.named_steps["preprocesador"]
            .named_transformers_["numericas"]
            .named_steps["imputador"]
        )
        self.assertEqual(imputador.statistics_[0], X_train["Age"].median())

    def test_pipeline_cargado_predice_con_categoria_desconocida(self) -> None:
        entrenar(DEFAULT_DATA_PATH, self.model_path)
        pasajero = {
            "Age": 29,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 25.0,
            "Pclass": 3,
            "Sex": "female",
            "Embarked": "X",
        }

        resultado = predecir([pasajero], self.model_path)
        self.assertEqual(len(resultado), 1)
        self.assertIn(resultado[0], {0, 1})

    def test_prediccion_rechaza_columnas_faltantes(self) -> None:
        entrenar(DEFAULT_DATA_PATH, self.model_path)

        with self.assertRaisesRegex(ValueError, "faltan columnas"):
            predecir([{"Age": 30}], self.model_path)


if __name__ == "__main__":
    unittest.main()
