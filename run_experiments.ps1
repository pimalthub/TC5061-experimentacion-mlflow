# Ejecuta las corridas del experimento en Windows (PowerShell).
# Uso:  .\run_experiments.ps1
$ErrorActionPreference = "Stop"
$EXP = "TC5061_RandomForest_CancerMama"

python train.py --experiment-name $EXP --run-name baseline
python train.py --experiment-name $EXP --run-name pocos_arboles     --n-estimators 10
python train.py --experiment-name $EXP --run-name muchos_arboles    --n-estimators 500
python train.py --experiment-name $EXP --run-name arboles_someros   --max-depth 2
python train.py --experiment-name $EXP --run-name profundidad_media --max-depth 5
python train.py --experiment-name $EXP --run-name hojas_grandes     --min-samples-leaf 20
python train.py --experiment-name $EXP --run-name todas_variables   --max-features 1.0
# Repetición exacta de la línea base para verificar reproducibilidad
python train.py --experiment-name $EXP --run-name baseline_repeticion
