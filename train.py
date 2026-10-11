"""
train.py — Entrenamiento reproducible de un clasificador Random Forest instrumentado con MLflow.

Problema: clasificar tumores de mama como malignos o benignos
(dataset Breast Cancer Wisconsin, incluido en scikit-learn).

Ejemplos de uso:
    python train.py                                   # valores por defecto
    python train.py --n-estimators 300 --max-depth 8 --min-samples-leaf 2 --seed 42

Ver resultados:
    mlflow ui --backend-store-uri sqlite:///mlflow.db   ->  http://127.0.0.1:5000
"""

import argparse
import inspect
import json
import os
import random
import tempfile
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: funciona en cualquier equipo
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRACKING_URI_DEFECTO = "sqlite:///mlflow.db"
EXPERIMENTO_DEFECTO = "TC5061_RandomForest_CancerMama"


# --------------------------------------------------------------------------------------
# 1. Argumentos de línea de comandos
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Entrena un Random Forest y registra la corrida en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Hiperparámetros del modelo
    p.add_argument("--n-estimators", type=int, default=200, help="Número de árboles del bosque")
    p.add_argument("--max-depth", type=int, default=None, help="Profundidad máxima de cada árbol (None = sin límite)")
    p.add_argument("--min-samples-leaf", type=int, default=1, help="Mínimo de muestras por hoja")
    p.add_argument("--max-features", default="sqrt",
                   help="Variables consideradas en cada división: 'sqrt', 'log2' o una fracción (ej. 0.5)")
    # Reproducibilidad y datos
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria global")
    p.add_argument("--test-size", type=float, default=0.2, help="Proporción del conjunto de prueba")
    # MLflow
    p.add_argument("--experiment-name", default=EXPERIMENTO_DEFECTO, help="Nombre del experimento en MLflow")
    p.add_argument("--run-name", default=None, help="Nombre de la corrida (si se omite se genera uno)")
    p.add_argument("--tracking-uri", default=TRACKING_URI_DEFECTO, help="Backend de MLflow")
    return p.parse_args(argv)


def normalizar_max_features(valor):
    """Convierte '0.5' -> 0.5 y deja 'sqrt'/'log2' como texto."""
    try:
        return float(valor)
    except (TypeError, ValueError):
        return valor


# --------------------------------------------------------------------------------------
# 2. Reproducibilidad
# --------------------------------------------------------------------------------------
def fijar_semilla(seed):
    """Fija todas las fuentes de aleatoriedad usadas por el script."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------------------
# 3. Datos
# --------------------------------------------------------------------------------------
def cargar_datos():
    datos = load_breast_cancer(as_frame=True)
    return datos.data, datos.target, list(datos.target_names)


def dividir_datos(X, y, test_size, seed):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# --------------------------------------------------------------------------------------
# 4. Modelo (preprocesamiento + clasificador)
# --------------------------------------------------------------------------------------
def construir_modelo(args):
    clasificador = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=normalizar_max_features(args.max_features),
        random_state=args.seed,
        n_jobs=1,  # un solo hilo: resultados idénticos en cualquier equipo
    )
    return Pipeline([("escalado", StandardScaler()), ("modelo", clasificador)])


def entrenar(modelo, X_train, y_train, seed):
    """Validación cruzada estratificada (5 folds) y ajuste final sobre todo el train."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_f1 = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="f1")
    modelo.fit(X_train, y_train)
    return modelo, cv_f1


# --------------------------------------------------------------------------------------
# 5. Evaluación
# --------------------------------------------------------------------------------------
def evaluar(modelo, X_test, y_test, cv_f1):
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "cv_f1_mean": float(cv_f1.mean()),
        "cv_f1_std": float(cv_f1.std()),
    }, y_pred


def generar_artefactos(modelo, X_test, y_test, y_pred, clases, metricas, carpeta):
    """Crea matriz de confusión, curva ROC, reporte de clasificación y metrics.json."""
    carpeta = Path(carpeta)
    rutas = []

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=clases,
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Matriz de confusión (prueba)")
    fig.tight_layout()
    rutas.append(carpeta / "confusion_matrix.png")
    fig.savefig(rutas[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(modelo, X_test, y_test, ax=ax, name="RandomForest")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("Curva ROC (prueba)")
    fig.tight_layout()
    rutas.append(carpeta / "roc_curve.png")
    fig.savefig(rutas[-1], dpi=120)
    plt.close(fig)

    rutas.append(carpeta / "classification_report.txt")
    rutas[-1].write_text(classification_report(y_test, y_pred, target_names=clases, digits=4),
                         encoding="utf-8")

    rutas.append(carpeta / "metrics.json")
    rutas[-1].write_text(json.dumps({k: round(v, 6) for k, v in metricas.items()}, indent=2),
                         encoding="utf-8")
    return rutas


def opciones_serializacion():
    """MLflow >= 3.x serializa con 'skops' y exige declarar el tipo Tree de scikit-learn.
    El modelo lo entrena este mismo script, por lo que es seguro marcarlo como confiable."""
    if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
        return {"skops_trusted_types": ["sklearn.tree._tree.Tree"]}
    return {}


# --------------------------------------------------------------------------------------
# 6. Orquestación con MLflow
# --------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    fijar_semilla(args.seed)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    X, y, clases = cargar_datos()
    X_train, X_test, y_train, y_test = dividir_datos(X, y, args.test_size, args.seed)

    run_name = args.run_name or (f"rf_n{args.n_estimators}_d{args.max_depth}"
                                 f"_leaf{args.min_samples_leaf}_mf{args.max_features}")

    with mlflow.start_run(run_name=run_name) as run:
        # Hiperparámetros y configuración
        parametros = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "seed": args.seed,
            "test_size": args.test_size,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        for nombre, valor in parametros.items():
            mlflow.log_param(nombre, valor)
        mlflow.set_tag("modelo", "RandomForestClassifier")
        mlflow.set_tag("dataset", "sklearn.load_breast_cancer")

        # Entrenamiento y evaluación
        modelo = construir_modelo(args)
        modelo, cv_f1 = entrenar(modelo, X_train, y_train, args.seed)
        metricas, y_pred = evaluar(modelo, X_test, y_test, cv_f1)
        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)

        # Artefactos
        with tempfile.TemporaryDirectory() as tmp:
            for ruta in generar_artefactos(modelo, X_test, y_test, y_pred, clases, metricas, tmp):
                mlflow.log_artifact(str(ruta), artifact_path="evaluacion")

        # Modelo entrenado (pipeline completo con firma)
        firma = infer_signature(X_train, modelo.predict(X_train))
        mlflow.sklearn.log_model(modelo, name="model", signature=firma,
                                 input_example=X_train.head(3), **opciones_serializacion())

        print(f"Corrida '{run_name}' (id {run.info.run_id})")
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metricas.items()))
    return metricas


if __name__ == "__main__":
    main()
