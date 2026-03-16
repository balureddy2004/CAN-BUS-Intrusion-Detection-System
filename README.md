## Running server (recommended env vars)

On PowerShell:
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"
python server.py

## Test predict
python test_predict_subprocess.py

## Inspect a model
python test_model_info.py
