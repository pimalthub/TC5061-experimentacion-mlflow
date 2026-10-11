# TC5061 · Experimentación reproducible con MLflow

Clasificador **Random Forest** para el dataset *Breast Cancer Wisconsin* (incluido en scikit-learn: maligno vs. benigno). El script es reutilizable, se parametriza desde la línea de comandos y registra cada corrida en **MLflow Tracking**.

## Estructura

```
├── train.py                      # carga de datos, preprocesamiento, entrenamiento, evaluación y registro en MLflow
├── run_experiments.sh / .ps1     # las 8 corridas del experimento (Linux-macOS / Windows)
├── notebooks/
│   └── 01_exploracion_datos.ipynb  # solo análisis exploratorio (EDA)
├── requirements.txt              # dependencias con versiones fijadas
└── README.md
```

## Instalación (Python ≥ 3.11)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

Con valores por defecto:

```bash
python train.py
```

Cambiando hiperparámetros sin editar el código:

```bash
python train.py --n-estimators 500 --max-depth 5 --min-samples-leaf 2 --max-features sqrt --seed 42
```

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--n-estimators` | 200 | Número de árboles |
| `--max-depth` | None | Profundidad máxima (None = sin límite) |
| `--min-samples-leaf` | 1 | Mínimo de muestras por hoja |
| `--max-features` | sqrt | Variables por división (`sqrt`, `log2` o fracción) |
| `--seed` | 42 | Semilla global (partición, CV y modelo) |
| `--test-size` | 0.2 | Proporción de prueba |
| `--experiment-name` | TC5061_RandomForest_CancerMama | Experimento de MLflow |
| `--run-name` | automático | Nombre de la corrida |

`python train.py --help` muestra la lista completa.

Para reproducir todas las corridas del informe:

```bash
bash run_experiments.sh        # Linux / macOS
.\run_experiments.ps1          # Windows PowerShell
```

## Ver y comparar las corridas

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre http://127.0.0.1:5000 → experimento **TC5061_RandomForest_CancerMama** → selecciona las corridas → **Compare**.

## Qué se registra en cada corrida

- **Parámetros** (`log_param`): `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `seed`, `test_size`, `n_train`, `n_test`.
- **Métricas** (`log_metric`): `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `cv_f1_mean`, `cv_f1_std`.
- **Artefactos** (`log_artifact`, carpeta `evaluacion/`): `confusion_matrix.png`, `roc_curve.png`, `classification_report.txt`, `metrics.json`.
- **Modelo** (`mlflow.sklearn.log_model`): pipeline completo (escalado + Random Forest) con firma de entrada/salida.

## Reproducibilidad

`train.py` fija la semilla en `random`, `numpy` y `PYTHONHASHSEED`, y la pasa como `random_state` a la partición estratificada, a la validación cruzada (`StratifiedKFold`) y al Random Forest. Además entrena con `n_jobs=1`. Por eso, **dos ejecuciones con los mismos argumentos producen métricas idénticas**, como lo muestran `baseline` y `baseline_repeticion` en el experimento.
